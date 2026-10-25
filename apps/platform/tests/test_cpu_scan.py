"""对齐原始扫描的输出与全部可训练输入梯度。"""
import pytest


@pytest.mark.parametrize("length", [1, 17, 129])
def test_cpu_scan_matches_reference_forward_and_backward(length):
    pytest.importorskip("numba")
    import torch
    from od_platform.backends.mixercseg.cpu_scan import selective_scan_cpu
    from od_platform.backends.mixercseg._vendor.VMamba.models.csms6s import selective_scan_torch
    torch.manual_seed(42)
    inputs = [torch.randn(2, 8, length), torch.randn(2, 8, length),
              -torch.rand(8, 3), torch.randn(2, 2, 3, length),
              torch.randn(2, 2, 3, length), torch.randn(8), torch.randn(8)]
    reference = [x.clone().requires_grad_() for x in inputs]
    accelerated = [x.clone().requires_grad_() for x in inputs]
    expected = selective_scan_torch(*reference)
    actual = selective_scan_cpu(*accelerated)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    gradient = torch.randn_like(actual)
    expected.backward(gradient); actual.backward(gradient)
    for old, new in zip(reference, accelerated, strict=True):
        torch.testing.assert_close(new.grad, old.grad, rtol=1e-4, atol=1e-4)
