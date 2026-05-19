#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Repaint (Chunk-consistent ODE INversion) implementation for LeRobot.

A training-free, gradient-free async inference method for flow matching robot
policies. Unlike RTC, which applies gradient-based velocity corrections at
every denoising step, Repaint finds the initial noise that satisfies the prefix
constraint via backward Euler inversion of the flow ODE.

Pipeline:
    1. Naive forward pass from x_0_free   ->  x_1_naive          [N model calls]
    2. Construct target: x_1_target = [prev_chunk[:d], x_1_naive[d:]]
    3. Backward Euler inversion          ->  x_0_star            [N model calls]
    4. Mao re-painting: x_0_repaint = [x_0_star[:d], x_0_free[d:]]
    5. Caller runs final forward from x_0_repaint                [N model calls]

Total cost: 3N model calls. No backpropagation. Compatible with TensorRT and
other inference-optimized runtimes where gradient computation is unavailable.

Reference:
    https://anonymous.4open.science/r/repaint
"""

import logging

import torch
from torch import Tensor

from lerobot.common.policies.repaint.configuration_repaint import RepaintConfig
from lerobot.common.policies.repaint.debug_tracker import Tracker

logger = logging.getLogger(__name__)


class RepaintProcessor:
    """Chunk-consistent ODE Inversion processor for flow matching policies.

    Computes the prefix-anchored initial noise x_0_repaint by inverting the
    flow ODE backward from a constructed target output. The caller then runs
    the standard forward ODE from x_0_repaint to produce a chunk that
    satisfies the prefix constraint without any velocity correction or
    gradient computation.
    """

    def __init__(self, repaint_config: RepaintConfig):
        self.repaint_config = repaint_config

        self.tracker = None

        if repaint_config.debug:
            self.tracker = Tracker(
                enabled=repaint_config.debug,
                maxlen=repaint_config.debug_maxlen,
            )

    # ====================== Tracker Proxy Methods ======================
    def track(
        self,
        phase: str,                                    # "forward" | "backward"
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x_0_free: Tensor | None = None,
        x_1_naive: Tensor | None = None,
        x_1_target: Tensor | None = None,
        x_0_star: Tensor | None = None,
        x_0_repaint: Tensor | None = None,
        inference_delay: int | None = None,
        **metadata,
    ) -> None:
        """Proxy method to track debug information.

        If tracker is None or disabled, this method does nothing.
        Otherwise, it forwards the call to tracker.track().
        """
        if self.tracker is not None:
            self.tracker.track(
                phase=phase,
                time=time,
                x_t=x_t,
                v_t=v_t,
                x_0_free=x_0_free,
                x_1_naive=x_1_naive,
                x_1_target=x_1_target,
                x_0_star=x_0_star,
                x_0_repaint=x_0_repaint,
                inference_delay=inference_delay,
                **metadata,
            )

    def get_all_debug_steps(self) -> list:
        """Get all debug steps from tracker.

        Returns empty list if tracker is disabled or None.
        """
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        """Check if debug tracking is enabled.

        Returns True if tracker exists and is enabled.
        """
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        """Reset the tracker, clearing all recorded steps.

        Does nothing if tracker is None.
        """
        if self.tracker is not None:
            self.tracker.reset()

    # ====================== End Tracker Proxy Methods ======================

    @torch.no_grad()
    def compute_repaint_noise(
        self,
        x_0_free: Tensor,
        prev_chunk_left_over: Tensor,
        inference_delay: int,
        original_denoise_step_partial,
        num_steps: int,
    ) -> Tensor:
        """Repaint inversion to find the prefix-anchored initial noise.

        Runs the flow ODE twice: once forward from fresh noise to obtain an
        on-manifold target, and once backward from that target to recover
        the noise that the model associates with the prefix. The returned
        noise is then passed to the caller's standard forward pass.

        Args:
            x_0_free (Tensor): Fresh Gaussian noise to seed the naive forward
                pass. Shape ``(B, T, A)``. Reused as the free-region noise in
                the Mao re-painting step.
            prev_chunk_left_over (Tensor): Unexecuted prefix from the previous
                chunk. Shape ``(B, T_prev, A)``. Specifies the values that the
                first ``inference_delay`` positions of the new chunk must
                produce. If ``None``, the method returns ``x_0_free`` unchanged.
            inference_delay (int): Number of prefix positions to anchor.
            original_denoise_step_partial (Callable[[Tensor, Tensor], Tensor]):
                Callable that computes the velocity given ``x_t`` and ``time``.
                Same signature as the inner denoiser used in sample_actions.
            num_steps (int): Number of Euler integration steps for the ODE.

        Returns:
            Tensor: ``x_0_repaint`` with the same shape as ``x_0_free``.
                Pass this to ``sample_actions(noise=x_0_repaint)`` for the
                final forward pass.

        Notes:
            - If ``prev_chunk_left_over`` is shorter than the new chunk length
              ``T``, it is right-padded with zeros to match ``T``.
            - The free region of the inverted noise ``x_0_star[d:]`` is
              discarded; we keep ``x_0_free[d:]`` instead, following the
              re-painting rule of Mao et al. [MM 2023].
            - This method does no backpropagation. The entire pipeline is
              compatible with TensorRT-compiled denoisers.

        Reference:
            https://anonymous.4open.science/r/repaint
        """
        if prev_chunk_left_over is None or inference_delay == 0:
            return x_0_free

        squeezed = False
        if x_0_free.ndim < 3:
            x_0_free = x_0_free.unsqueeze(0)
            squeezed = True

        if prev_chunk_left_over.ndim < 3:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        batch_size, action_chunk_size, action_dim = x_0_free.shape

        # Right-pad prev_chunk_left_over to match the new chunk length if needed
        if (prev_chunk_left_over.shape[1] < action_chunk_size
                or prev_chunk_left_over.shape[2] < action_dim):
            padded = torch.zeros(
                batch_size, action_chunk_size, action_dim,
                dtype=x_0_free.dtype, device=x_0_free.device,
            )
            padded[
                :,
                : prev_chunk_left_over.shape[1],
                : prev_chunk_left_over.shape[2],
            ] = prev_chunk_left_over
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_0_free.shape, (
            "The padded previous chunk must match the input noise shape"
        )

        # prefix_mask: True for [:d], False for [d:] — broadcastable [1, T, 1]
        prefix_mask = (
            torch.arange(action_chunk_size, device=x_0_free.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            < inference_delay
        )

        # ── Step 1: naive forward ODE  -->  x_1_naive ─────────────────────
        x_1_naive = self._forward_ode(
            x_0=x_0_free,
            original_denoise_step_partial=original_denoise_step_partial,
            num_steps=num_steps,
        )

        # ── Step 2: construct inversion target ────────────────────────────
        x_1_target = torch.where(prefix_mask, prev_chunk_left_over, x_1_naive)

        # ── Step 3: backward Euler inversion  -->  x_0_star ───────────────
        x_0_star = self._backward_ode(
            x_1_target=x_1_target,
            original_denoise_step_partial=original_denoise_step_partial,
            num_steps=num_steps,
        )

        # ── Step 4: Mao re-painting  -->  x_0_repaint ─────────────────────
        # Keep x_0_free[d:] (obs-consistent), not fresh noise (uncorrelated)
        x_0_repaint = torch.where(prefix_mask, x_0_star, x_0_free)

        self.track(
            phase="repaint",
            time=0.0,
            x_0_free=x_0_free,
            x_1_naive=x_1_naive,
            x_1_target=x_1_target,
            x_0_star=x_0_star,
            x_0_repaint=x_0_repaint,
            inference_delay=inference_delay,
        )

        if squeezed:
            x_0_repaint = x_0_repaint.squeeze(0)

        return x_0_repaint

    # ====================== ODE Integration Helpers ======================

    @torch.no_grad()
    def _forward_ode(
        self,
        x_0: Tensor,
        original_denoise_step_partial,
        num_steps: int,
    ) -> Tensor:
        """Run the standard forward ODE from t=1 to t=0  [N model calls].

        Follows the same time convention as ``sample_actions``: time starts
        at 1.0 and decreases by ``dt = -1/N`` at each step until t<=0.
        """
        bsize = x_0.shape[0]
        device = x_0.device

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = x_0.clone()
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = original_denoise_step_partial(x_t, expanded_time)
            x_t = x_t + dt * v_t

            self.track(phase="forward", time=time, x_t=x_t, v_t=v_t)

            time = time + dt

        return x_t

    @torch.no_grad()
    def _backward_ode(
        self,
        x_1_target: Tensor,
        original_denoise_step_partial,
        num_steps: int,
    ) -> Tensor:
        """Run the inverse ODE from t=0 (data) to t=1 (noise)  [N model calls].

        Sign of dt is flipped relative to ``_forward_ode``. Time increases
        from 0.0 to 1.0 along the backward trajectory, with each step using
        the model's velocity at the current state to integrate in reverse.
        """
        bsize = x_1_target.shape[0]
        device = x_1_target.device

        dt = torch.tensor(1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = x_1_target.clone()
        time = torch.tensor(0.0, dtype=torch.float32, device=device)

        while time <= 1.0 - dt / 2:
            expanded_time = time.expand(bsize)
            v_t = original_denoise_step_partial(x_t, expanded_time)
            x_t = x_t + dt * v_t

            self.track(phase="backward", time=time, x_t=x_t, v_t=v_t)

            time = time + dt

        return x_t