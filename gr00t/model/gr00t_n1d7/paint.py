# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PAINT: prefix-consistent action chunking for flow-matching policies.

Under asynchronous execution a new chunk arrives while the previous one is still
running, so its first ``inference_delay`` steps are already committed on the
robot and must agree with what is being executed.

``repaint_sample`` (PAINT) pins that prefix by inverting the learned flow: run a
free forward pass, substitute the committed prefix into the clean sample,
integrate backwards to recover the noise producing that prefix, then re-run the
forward pass from noise whose prefix is that recovered noise and whose suffix is
the original free noise. Three ODE sweeps, no gradients.

``rtc_sample`` (real-time chunking) is the guidance baseline: each Euler step is
steered by a VJP of the one-step denoiser. One forward + one VJP per step, and it
does need autograd.

Both take a ``velocity_fn`` closure rather than a model, so they stay independent
of how the head is conditioned.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch


PREFIX_SCHEDULES = ("ones", "zeros", "linear", "exp")


class VelocityFn(Protocol):
    """``(actions [B, H, D], t_cont in [0, 1)) -> velocity [B, H, D]``."""

    def __call__(self, actions: torch.Tensor, t_cont: float) -> torch.Tensor: ...


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-timestep weight on the previous chunk across an action horizon.

    With ``start=2, end=6, total=10`` the output is::

        1  1  4/5 3/5 2/5 1/5 0  0  0  0
              ^              ^
            start           end

    ``start`` (inclusive) is where the chunk starts being allowed to change,
    ``end`` (exclusive) where it stops attending to the prefix entirely.

    ``end`` takes precedence: if ``end < start`` then ``start`` is pushed down to
    ``end``, so ``end == 0`` always ignores the prefix.
    """
    if schedule not in PREFIX_SCHEDULES:
        raise ValueError(f"Invalid schedule: {schedule!r}. Expected one of {PREFIX_SCHEDULES}.")
    start = min(start, end)
    idx = torch.arange(total, dtype=torch.float32, device=device)
    if schedule == "ones":
        w = torch.ones(total, dtype=torch.float32, device=device)
    elif schedule == "zeros":
        w = (idx < start).float()
    else:  # "linear" or "exp"
        w = torch.clamp((start - 1 - idx) / (end - start + 1) + 1, min=0, max=1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1.0)
    w = torch.where(idx >= end, torch.zeros((), dtype=w.dtype, device=w.device), w)
    return w.to(dtype=dtype)


def _prefix_mask(horizon: int, inference_delay: int, device: torch.device) -> torch.Tensor:
    return torch.arange(horizon, device=device).view(1, -1, 1) < inference_delay


def _integrate(
    velocity_fn: VelocityFn, x: torch.Tensor, num_steps: int, *, reverse: bool = False
) -> torch.Tensor:
    """Euler-integrate the flow ODE across [0, 1].

    The reverse pass evaluates the velocity at the arriving rather than the
    departing point — the explicit inversion PAINT relies on, accurate to O(dt).
    """
    dt = 1.0 / num_steps
    steps = reversed(range(num_steps)) if reverse else range(num_steps)
    for step in steps:
        v = velocity_fn(x, step / float(num_steps))
        x = x - dt * v if reverse else x + dt * v
    return x


def repaint_sample(
    velocity_fn: VelocityFn,
    x_0_free: torch.Tensor,
    prev_action_chunk: torch.Tensor,
    *,
    inference_delay: int,
    num_steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Sample an action chunk whose prefix matches the committed actions.

    ``prev_action_chunk`` must already be time-shifted so index 0 is the next
    action to execute; only its first ``inference_delay`` steps are read.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    horizon = x_0_free.shape[1]
    if not 0 <= inference_delay <= horizon:
        raise ValueError(
            f"inference_delay must be in [0, {horizon}] for an action horizon of "
            f"{horizon}, got {inference_delay}"
        )

    # Step 1: free forward pass — the unconstrained plan.
    x_1_naive = _integrate(velocity_fn, x_0_free, num_steps)

    # With nothing committed, steps 2-5 provably reduce to the identity. Short
    # circuit so the caller is not billed for two sweeps that cannot change the
    # answer, and record it so the cost is auditable.
    if inference_delay == 0:
        return x_1_naive, {"repaint_applied": False, "prefix_error": 0.0}

    prefix_mask = _prefix_mask(horizon, inference_delay, x_0_free.device)
    prev_action_chunk = prev_action_chunk.to(device=x_0_free.device, dtype=x_0_free.dtype)

    # Step 2: pin the committed prefix onto the clean sample.
    x_1_target = torch.where(prefix_mask, prev_action_chunk, x_1_naive)

    # Step 3: invert the flow to recover the noise producing that prefix.
    x_0_star = _integrate(velocity_fn, x_1_target, num_steps, reverse=True)

    # Step 4: repaint — recovered noise on the prefix, free noise elsewhere, so
    # only the prefix is constrained and the suffix is re-planned freely.
    x_0_repaint = torch.where(prefix_mask, x_0_star, x_0_free)

    # Step 5: final forward pass from the repainted noise.
    x_1 = _integrate(velocity_fn, x_0_repaint, num_steps)

    prefix_error = (
        (x_1[:, :inference_delay] - prev_action_chunk[:, :inference_delay]).abs().mean().item()
    )
    return x_1, {"repaint_applied": True, "prefix_error": prefix_error}


def rtc_sample(
    velocity_fn: VelocityFn,
    x_0: torch.Tensor,
    prev_action_chunk: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_steps: int,
    max_guidance_weight: float,
    sigma_d_o: float,
    action_dim_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Guide each Euler step toward the committed prefix.

    The one-step denoiser ``x -> x + v(x, t) * dt`` is linearized with a VJP and
    nudged so its clean-sample estimate moves toward ``prev_action_chunk`` on the
    weighted prefix. Requires grad mode: callers must not use ``inference_mode``.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    dt = 1.0 / num_steps
    x_t = x_0
    prev_action_chunk = prev_action_chunk.to(device=x_0.device, dtype=x_0.dtype)
    weights = weights.to(device=x_0.device, dtype=x_0.dtype)

    with torch.enable_grad():
        for step in range(num_steps):
            t_cont = step / float(num_steps)

            def denoiser(x: torch.Tensor, _t: float = t_cont) -> tuple[torch.Tensor, torch.Tensor]:
                v = velocity_fn(x, _t)
                return x + v * dt, v

            (x_1_hat, v_t), vjp_fn = torch.func.vjp(denoiser, x_t)

            error = (prev_action_chunk - x_1_hat) * weights[:, None]
            if action_dim_mask is not None:
                error = error * action_dim_mask.to(device=x_0.device, dtype=x_0.dtype)

            # Pull the prefix error back through the denoiser only; the velocity
            # output carries no target, hence the zero cotangent.
            correction = vjp_fn((error, torch.zeros_like(v_t)))[0]
            if correction is None:
                correction = torch.zeros_like(x_t)

            guidance_weight = _rtc_guidance_weight(t_cont, sigma_d_o, max_guidance_weight)
            v_corrected = v_t + guidance_weight * correction
            # Detach between steps: guidance is per-step, so retaining the graph
            # would grow memory linearly in num_steps for no benefit. The VJP is
            # unaffected, as torch.func.vjp treats its primal as a leaf.
            x_t = (x_t + v_corrected * dt).detach()

    return x_t, {
        "repaint_applied": True,
        "prefix_error": _weighted_prefix_error(x_t, prev_action_chunk, weights),
    }


def _rtc_guidance_weight(t_cont: float, sigma_d_o: float, max_guidance_weight: float) -> float:
    """Guidance coefficient ``c(t) / r(t)^2``.

    ``c(t) = (1 - t) / t`` diverges at ``t = 0`` and ``1 / r(t)^2`` at ``t = 1``;
    both ends clamp to ``max_guidance_weight``.
    """
    one_minus_t = 1.0 - t_cont
    denom = (sigma_d_o**2) * (one_minus_t**2)
    if denom <= 0.0 or t_cont <= 0.0:
        return max_guidance_weight
    inv_r2 = ((sigma_d_o**2) * (t_cont**2) + one_minus_t**2) / denom
    weight = (one_minus_t / t_cont) * inv_r2
    if not math.isfinite(weight):
        return max_guidance_weight
    return min(weight, max_guidance_weight)


def _weighted_prefix_error(
    x: torch.Tensor, prev_action_chunk: torch.Tensor, weights: torch.Tensor
) -> float:
    """Mean absolute deviation over the timesteps the weights actually pin."""
    pinned = weights > 0
    if not bool(pinned.any()):
        return 0.0
    return (x[:, pinned] - prev_action_chunk[:, pinned]).abs().mean().item()


def blend_prefix(
    x: torch.Tensor, prev_action_chunk: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """``prev * w + x * (1 - w)``, so the executed prefix is exactly continuous."""
    weights = weights.to(device=x.device, dtype=x.dtype)
    prev_action_chunk = prev_action_chunk.to(device=x.device, dtype=x.dtype)
    return prev_action_chunk * weights[:, None] + x * (1 - weights[:, None])


def make_action_dim_mask(
    reference: torch.Tensor, actual_action_dim: int | None
) -> torch.Tensor | None:
    """[B, H, D] mask keeping only the embodiment's real action dims.

    Checkpoints pad actions out to ``max_action_dim``; guiding on the padded tail
    would spend the guidance budget on dimensions nothing executes.
    """
    if actual_action_dim is None:
        return None
    dim = reference.shape[-1]
    if not 0 < actual_action_dim <= dim:
        raise ValueError(
            f"actual_action_dim must be in (0, {dim}] for an action dim of {dim}, "
            f"got {actual_action_dim}"
        )
    if actual_action_dim == dim:
        return None
    mask = torch.zeros_like(reference)
    mask[..., :actual_action_dim] = 1.0
    return mask


def shift_action_chunk(prev_action_chunk: torch.Tensor, execution_horizon: int) -> torch.Tensor:
    """Roll a chunk forward so index 0 is the next action due on the robot.

    The vacated tail is zero-filled; those entries always carry weight 0 under
    :func:`get_prefix_weights` and lie outside the repaint prefix, so they are
    never read back.
    """
    if execution_horizon <= 0:
        raise ValueError(f"execution_horizon must be >= 1, got {execution_horizon}")
    horizon = prev_action_chunk.shape[1]
    if execution_horizon >= horizon:
        return torch.zeros_like(prev_action_chunk)
    tail = torch.zeros(
        (prev_action_chunk.shape[0], execution_horizon, prev_action_chunk.shape[-1]),
        device=prev_action_chunk.device,
        dtype=prev_action_chunk.dtype,
    )
    return torch.cat((prev_action_chunk[:, execution_horizon:], tail), dim=1)


def resolve_prefix_attention_horizon(
    prefix_attention_horizon: int | None, action_horizon: int, execution_horizon: int
) -> int:
    """Default the blend cutoff to the part of the chunk that survives execution."""
    if prefix_attention_horizon is not None:
        return prefix_attention_horizon
    return max(action_horizon - execution_horizon, 0)


__all__ = [
    "PREFIX_SCHEDULES",
    "VelocityFn",
    "blend_prefix",
    "get_prefix_weights",
    "make_action_dim_mask",
    "repaint_sample",
    "resolve_prefix_attention_horizon",
    "rtc_sample",
    "shift_action_chunk",
]
