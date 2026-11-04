"""The 6.13 weight edit as a LoRA adapter (docs/rive-design.md section 6.15). The edit W <- W - r r^T W changes each
writer matrix by rank 1, so it is exactly the LoRA delta B A with B = -r and A = r^T W (rank 1, alpha 1). As an
adapter it can be switched on per request, on top of the original model: the thinking requests get it, the
non-thinking ones (which it harms, 6.14) do not.
  python scripts/rive_edit_lora.py e1     # writes ~/alter/models/lora-nowait-<name> (PEFT format)
Needs the direction from scripts/rive_think_fix.py (~/alter/rive/thinkfix/direction_<name>.pt)."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rive_think_fix as fix  # noqa: E402


def write(name, thinkfix=Path.home() / "alter/rive/thinkfix"):
    import torch
    from safetensors.torch import load_file, save_file
    r = torch.load(thinkfix / f"direction_{name}.pt").to(torch.float32)  # on the CPU: the server keeps the GPU
    target = fix.MODEL.parent / f"lora-nowait-{name}"
    target.mkdir(exist_ok=True)
    tensors = {}
    for f in sorted(fix.MODEL.glob("*.safetensors")):
        for key, w in load_file(f).items():
            if key.startswith("model.language_model.layers.") and key.endswith(fix.WRITERS):
                module = key.removesuffix(".weight")
                a = r @ w.to(torch.float32)  # (in,)
                tensors[f"base_model.model.{module}.lora_A.weight"] = a[None].to(w.dtype).cpu().contiguous()
                tensors[f"base_model.model.{module}.lora_B.weight"] = (-r)[:, None].to(w.dtype).cpu().contiguous()
    save_file(tensors, target / "adapter_model.safetensors", metadata={"format": "pt"})
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 1, "lora_alpha": 1, "lora_dropout": 0.0,
              "bias": "none", "fan_in_fan_out": False, "base_model_name_or_path": str(fix.MODEL),
              "target_modules": ["down_proj", "o_proj", "out_proj"]}
    (target / "adapter_config.json").write_text(json.dumps(config, indent=1))
    print(f"{time.strftime('%F %T')} lora {name}: {len(tensors) // 2} matrices, {target}", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["e1"]:
        write(n)
