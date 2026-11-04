"""Weight edits as low-rank changes (docs/rive-design.md section 6.23).

Every edit of one matrix W (out x in, as stored in the checkpoint) is a pair of factors A (k x in) and B (out x k),
so that W changes by B @ A. Parts of an edit that touch the same matrix are stacked, and the whole edit is written
as a LoRA adapter that vLLM loads at run time (section 6.15). The kinds:

- write_project:  what W writes along direction r (unit, out-space) is scaled by (1 - alpha): B = -alpha r, A = r^T W.
- read_project:   what W reads along direction r (unit, in-space) is scaled by (1 - alpha): A = r^T, B = -alpha W r.
- write_redirect: a share alpha of what W writes along r is moved onto s: A = r^T W, B = alpha (s - r).
- read_offset:    W reads as if the residual had beta * v added, using the mean input direction m (unit) of the
                  normalised residual: A = m^T, B = beta * W v / m_proj, where m_proj is the mean of m^T x. This is
                  exact only for inputs whose projection on m equals the mean; the caller reports how much it varies.
- singular_scale: the top k singular components of W are scaled by gamma: A = S_k V_k^T, B = (gamma - 1) U_k.
- neuron_scale:   input columns idx of W (MLP neurons, for down_proj) are scaled by gamma: A = one-hot rows,
                  B = (gamma - 1) W[:, idx]. Exact.
- head_scale:     the input columns of one head are scaled by gamma, approximated by the head's top k singular
                  components (a head has more columns than the rank allows); the share of the head's energy kept
                  is reported.
"""
import json
from pathlib import Path

import torch

MAX_RANK = 64
# vLLM fuses these checkpoint matrices into one layer; an adapter that changes one of them gets zero factors for
# the others, so the fused layer is always complete.
PACKED = [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), ("in_proj_qkv", "in_proj_z"),
          ("in_proj_b", "in_proj_a")]


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    n = v.norm()
    if n == 0:
        raise ValueError("a zero vector has no direction")
    return v / n


def write_project(w, r, alpha):
    r = unit(r)
    return (r @ w)[None, :], (-float(alpha) * r)[:, None]


def read_project(w, r, alpha):
    r = unit(r)
    return r[None, :], (-float(alpha) * (w @ r))[:, None]


def write_redirect(w, r, s, alpha):
    r, s = unit(r), unit(s)
    return (r @ w)[None, :], (float(alpha) * (s - r))[:, None]


def read_offset(w, v, m, m_proj, beta):
    if abs(m_proj) < 1e-6:
        raise ValueError("the mean input has no length along m")
    v = torch.as_tensor(v, dtype=torch.float32)
    return unit(m)[None, :], (float(beta) * (w @ v) / float(m_proj))[:, None]


def top_singular(w, k):
    """The top k singular triples: exact for small matrices, randomized (6 power iterations) for large ones."""
    if min(w.shape) <= 1024:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
    else:
        u, s, v = torch.svd_lowrank(w, q=min(k + 10, min(w.shape)), niter=6)
        vh = v.T
    return u[:, :k], s[:k], vh[:k]


def singular_scale(w, k, gamma):
    u, s, vh = top_singular(w, k)
    return s[:, None] * vh, (float(gamma) - 1.0) * u


def neuron_scale(w, idx, gamma):
    idx = sorted({int(i) for i in idx})
    if not idx or idx[0] < 0 or idx[-1] >= w.shape[1]:
        raise ValueError(f"neuron indices must lie in 0..{w.shape[1] - 1}")
    a = torch.zeros(len(idx), w.shape[1])
    a[torch.arange(len(idx)), torch.tensor(idx)] = 1.0
    return a, (float(gamma) - 1.0) * w[:, idx]


def head_scale(w, head, head_dim, gamma, k):
    lo = int(head) * int(head_dim)
    if lo < 0 or lo + head_dim > w.shape[1]:
        raise ValueError(f"head must lie in 0..{w.shape[1] // head_dim - 1}")
    block = w[:, lo:lo + head_dim]
    u, s, vh = top_singular(block, min(k, head_dim))
    a = torch.zeros(len(s), w.shape[1])
    a[:, lo:lo + head_dim] = s[:, None] * vh
    kept = float((s ** 2).sum() / (block ** 2).sum())
    return a, (float(gamma) - 1.0) * u, kept


def stack(factors):
    """Parts on one matrix -> one (A, B); their ranks add up."""
    a = torch.cat([f[0] for f in factors], 0)
    b = torch.cat([f[1] for f in factors], 1)
    if a.shape[0] > MAX_RANK:
        raise ValueError(f"the parts on one matrix add up to rank {a.shape[0]}, more than {MAX_RANK}")
    return a, b


def relative_size(w, a, b):
    """|B A| / |W| in the Frobenius norm, without forming B A."""
    return float(torch.sqrt(torch.trace((b.T @ b) @ (a @ a.T)).clamp(min=0)) / w.norm())


def export_lora(target, factors, shapes, base_model, dtype=torch.bfloat16):
    """factors: {checkpoint key ("model.language_model.layers.N.....weight"): (A, B)}; shapes: {key: (out, in)} for
    every matrix of the model, used for the zero factors of fused partners. Writes a PEFT adapter vLLM can load."""
    from safetensors.torch import save_file
    rank = max(a.shape[0] for a, _ in factors.values())
    keys = set(factors)
    for key in list(factors):
        stem, name = key.removesuffix(".weight").rsplit(".", 1)
        for group in PACKED:
            if name in group:
                keys |= {f"{stem}.{other}.weight" for other in group if f"{stem}.{other}.weight" in shapes}
    tensors = {}
    for key in sorted(keys):
        out, inp = shapes[key]
        a, b = factors.get(key, (torch.zeros(0, inp), torch.zeros(out, 0)))
        pa, pb = torch.zeros(rank, inp), torch.zeros(out, rank)
        pa[:a.shape[0]], pb[:, :b.shape[1]] = a, b
        module = key.removesuffix(".weight")
        tensors[f"base_model.model.{module}.lora_A.weight"] = pa.to(dtype).contiguous()
        tensors[f"base_model.model.{module}.lora_B.weight"] = pb.to(dtype).contiguous()
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    save_file(tensors, target / "adapter_model.safetensors", metadata={"format": "pt"})
    (target / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": rank, "lora_dropout": 0.0,
         "bias": "none", "fan_in_fan_out": False, "base_model_name_or_path": str(base_model),
         "target_modules": sorted({k.removesuffix(".weight").rsplit(".", 1)[1] for k in keys})}))
    return {"rank": rank, "matrices": len(factors), "with_fused_partners": len(keys)}
