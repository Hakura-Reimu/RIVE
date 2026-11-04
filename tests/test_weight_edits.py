import json

import pytest
import torch
from safetensors.torch import load_file

from rive.meta_rsi import weight_edits as we

torch.manual_seed(0)
W = torch.randn(12, 20)


def delta(a, b):
    return b @ a


def test_write_project_removes_what_is_written_along_r():
    r = we.unit(torch.randn(12))
    a, b = we.write_project(W, r, 1.0)
    new = W + delta(a, b)
    assert torch.allclose(r @ new, torch.zeros(20), atol=1e-5)
    half = W + delta(*we.write_project(W, r, 0.5))
    assert torch.allclose(r @ half, 0.5 * (r @ W), atol=1e-5)


def test_read_project_makes_w_blind_to_r():
    r = we.unit(torch.randn(20))
    new = W + delta(*we.read_project(W, r, 1.0))
    assert torch.allclose(new @ r, torch.zeros(12), atol=1e-5)


def test_write_redirect_moves_the_component_onto_s():
    r, s = we.unit(torch.randn(12)), we.unit(torch.randn(12))
    new = W + delta(*we.write_redirect(W, r, s, 1.0))
    x = torch.randn(20)
    assert torch.allclose(new @ x, W @ x + (s - r) * (r @ (W @ x)), atol=1e-5)


def test_read_offset_is_exact_for_an_input_at_the_mean_projection():
    v, m = torch.randn(20), we.unit(torch.randn(20))
    x = 3.0 * m  # its projection on m is 3
    a, b = we.read_offset(W, v, m, 3.0, 0.5)
    assert torch.allclose((W + delta(a, b)) @ x, W @ (x + 0.5 * v), atol=1e-4)


def test_singular_scale_scales_the_top_components():
    a, b = we.singular_scale(W, 3, 0.0)
    s_new = torch.linalg.svdvals(W + delta(a, b))
    s_old = torch.linalg.svdvals(W)
    assert torch.allclose(s_new[:9], s_old[3:12], atol=1e-4)


def test_neuron_scale_is_exact():
    new = W + delta(*we.neuron_scale(W, [2, 5], 0.0))
    assert torch.all(new[:, [2, 5]] == 0) and torch.allclose(new[:, 3], W[:, 3])
    with pytest.raises(ValueError):
        we.neuron_scale(W, [20], 0.0)


def test_head_scale_is_exact_when_the_rank_covers_the_head():
    a, b, kept = we.head_scale(W, 1, 5, 0.0, 5)
    new = W + delta(a, b)
    assert kept == pytest.approx(1.0) and torch.allclose(new[:, 5:10], torch.zeros(12, 5), atol=1e-5)
    assert torch.allclose(new[:, :5], W[:, :5]) and torch.allclose(new[:, 10:], W[:, 10:])
    assert we.head_scale(W, 1, 5, 0.0, 2)[2] < 1.0


def test_stack_adds_ranks_and_refuses_too_many():
    r = we.unit(torch.randn(12))
    a, b = we.stack([we.write_project(W, r, 1.0), we.singular_scale(W, 2, 0.5)])
    assert a.shape == (3, 20) and b.shape == (12, 3)
    with pytest.raises(ValueError):
        we.stack([we.write_project(W, r, 1.0)] * 65)


def test_export_names_pads_and_completes_fused_layers(tmp_path):
    key = "model.language_model.layers.3.self_attn.q_proj.weight"
    shapes = {key: (12, 20), "model.language_model.layers.3.self_attn.k_proj.weight": (6, 20),
              "model.language_model.layers.3.self_attn.v_proj.weight": (6, 20),
              "model.language_model.layers.3.mlp.down_proj.weight": (20, 12)}
    r = we.unit(torch.randn(20))
    down = torch.randn(20, 12)
    factors = {key: we.read_project(W, r, 1.0),
               "model.language_model.layers.3.mlp.down_proj.weight": we.singular_scale(down, 2, 0.5)}
    info = we.export_lora(tmp_path / "e", factors, shapes, "/models/x", dtype=torch.float32)
    assert info == {"rank": 2, "matrices": 2, "with_fused_partners": 4}
    t = load_file(tmp_path / "e" / "adapter_model.safetensors")
    pre = "base_model.model.model.language_model.layers.3."
    assert t[pre + "self_attn.q_proj.lora_A.weight"].shape == (2, 20)
    assert t[pre + "self_attn.k_proj.lora_B.weight"].abs().sum() == 0  # the fused partner is a zero change
    got = t[pre + "self_attn.q_proj.lora_B.weight"] @ t[pre + "self_attn.q_proj.lora_A.weight"]
    assert torch.allclose(got, delta(*factors[key]), atol=1e-6)
    cfg = json.loads((tmp_path / "e" / "adapter_config.json").read_text())
    assert cfg["r"] == cfg["lora_alpha"] == 2 and cfg["target_modules"] == ["down_proj", "k_proj", "q_proj", "v_proj"]


def test_relative_size_matches_the_formed_change():
    a, b = we.singular_scale(W, 2, 0.5)
    assert we.relative_size(W, a, b) == pytest.approx(float(delta(a, b).norm() / W.norm()), rel=1e-4)
