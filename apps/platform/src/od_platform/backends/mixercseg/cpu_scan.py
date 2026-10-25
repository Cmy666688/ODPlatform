"""CPU 扫描加速:与上游相同的逐步递推及解析反传,不启用 fastmath。"""
from __future__ import annotations

import numpy as np
import torch
from numba import njit


@njit(cache=True, nogil=True)
def _forward(a, b, c):
    batch, channels, length, states = a.shape
    hidden = np.zeros_like(a)
    output = np.zeros((batch, channels, length), dtype=np.float32)
    for n in range(batch):
        for channel in range(channels):
            for t in range(length):
                value = np.float32(0)
                for s in range(states):
                    previous = hidden[n, channel, t - 1, s] if t else np.float32(0)
                    h = a[n, channel, t, s] * previous + b[n, channel, t, s]
                    hidden[n, channel, t, s] = h
                    value += h * c[n, channel, t, s]
                output[n, channel, t] = value
    return output, hidden


@njit(cache=True, nogil=True)
def _backward(a, hidden, c, grad):
    batch, channels, length, states = a.shape
    ga, gb, gc = np.zeros_like(a), np.zeros_like(a), np.zeros_like(a)
    for n in range(batch):
        for channel in range(channels):
            carry = np.zeros(states, dtype=np.float32)
            for t in range(length - 1, -1, -1):
                for s in range(states):
                    g = carry[s] + grad[n, channel, t] * c[n, channel, t, s]
                    previous = hidden[n, channel, t - 1, s] if t else np.float32(0)
                    ga[n, channel, t, s] = g * previous
                    gb[n, channel, t, s] = g
                    gc[n, channel, t, s] = grad[n, channel, t] * hidden[n, channel, t, s]
                    carry[s] = g * a[n, channel, t, s]
    return ga, gb, gc


class _Recurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c):
        arrays = [x.detach().contiguous().numpy() for x in (a, b, c)]
        output, hidden = _forward(*arrays)
        ctx.arrays = arrays[0], hidden, arrays[2]
        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad):
        gradients = _backward(*ctx.arrays, grad.detach().contiguous().numpy())
        return tuple(torch.from_numpy(g) for g in gradients)


def selective_scan_cpu(u, delta, A, B, C, D=None, delta_bias=None,
                       delta_softplus=True, oflex=True, *args, **kwargs):
    if u.device.type != "cpu":
        raise ValueError("CPU JIT 扫描仅支持 CPU")
    dtype = u.dtype
    batch, groups, states, length = B.shape
    channels = u.shape[1]
    per_group = channels // groups
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    u, delta, A, B, C = [x.float() for x in (u, delta, A, B, C)]
    B = B.view(batch, groups, 1, states, length).repeat(1, 1, per_group, 1, 1).view(batch, channels, states, length)
    C = C.view(batch, groups, 1, states, length).repeat(1, 1, per_group, 1, 1).view(batch, channels, states, length)
    a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    b = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    output = _Recurrence.apply(a, b, C.permute(0, 1, 3, 2))
    output = output if D is None else output + u * D.unsqueeze(-1)
    return output if oflex else output.to(dtype=dtype)


def configure_cpu_scan(enabled: bool) -> None:
    from ._vendor.VMamba.models import csms6s, vmamba
    vmamba.selective_scan_fn = selective_scan_cpu if enabled else csms6s.selective_scan_fn
