"""上游扫描算子的 CPU 梯度与 HOG checkpoint 结构兼容。"""
import pytest


def test_torch_scan_merge_preserves_orientation_and_gradient():
    import torch
    from od_platform.backends.mixercseg._vendor.VMamba.models.csm_triton import cross_scan_fn, cross_merge_fn
    image = torch.arange(12.0).reshape(1, 1, 3, 4).requires_grad_()
    scanned = cross_scan_fn(image)
    merged = cross_merge_fn(scanned.reshape(1, 4, 1, 3, 4)).reshape_as(image)
    assert torch.equal(merged, image * 4)
    merged.sum().backward()
    assert torch.equal(image.grad, torch.full_like(image, 4))


def test_selective_scan_reference_has_recurrence_gradient():
    import torch
    from od_platform.backends.mixercseg._vendor.VMamba.models.csms6s import selective_scan_torch
    values = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    output = selective_scan_torch(values, torch.ones_like(values), torch.zeros(1, 1),
                                   torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3),
                                   delta_softplus=False)
    assert torch.allclose(output, torch.tensor([[[1.0, 3.0, 6.0]]]))
    output.sum().backward()
    assert torch.allclose(values.grad, torch.tensor([[[3.0, 2.0, 1.0]]]))


def test_published_hog_variant_has_checkpoint_parameter_shapes():
    pytest.importorskip("timm")
    pytest.importorskip("fvcore")
    from od_platform.backends.mixercseg.model import build_model
    from od_platform.runtime_config.mixercseg import MixerCSegConfig
    model, _criterion, device = build_model(MixerCSegConfig(hog_bins=180, device="cpu"))
    assert device == "cpu"
    weights = model.state_dict()
    assert weights["backbone.vss_layers.0.blocks.0.1.hog_feat.0.weight"].shape == (16, 180, 1, 1)
    assert sum(p.numel() for p in model.parameters()) == 2541882


def test_cpu_dispatch_does_not_call_installed_cuda_extensions(monkeypatch):
    import torch
    from od_platform.backends.mixercseg._vendor.VMamba.models import csms6s
    monkeypatch.setattr(csms6s, "WITH_SELECTIVESCAN_OFLEX", True)
    values = torch.tensor([[[1.0, 2.0, 3.0]]])
    output = csms6s.selective_scan_fn(values, torch.ones_like(values), torch.zeros(1, 1),
                                     torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3),
                                     delta_softplus=False, backend="oflex")
    assert torch.allclose(output, torch.tensor([[[1.0, 3.0, 6.0]]]))
