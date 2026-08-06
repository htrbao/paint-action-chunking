# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
TensorRT backend for repaint-euler.

Repaint-euler is gradient-free: it only ever evaluates the denoiser forward, three ODE
passes per chunk (free forward -> backward inversion -> repainted forward). That is what
makes it TensorRT-deployable, and what `--smooth-option rtc` is not — RTC differentiates
through the denoiser with `torch.func.vjp`, and a TensorRT engine has no backward.

No new engines are required. This reuses exactly the ones `export_onnx.py` already
builds, and sweeps them three times per chunk instead of once:

    vlln_vl_self_attention, state_encoder, action_encoder, DiT, action_decoder

Usage
-----
    from deployment_scripts.trt_repaint_forward import setup_tensorrt_engines_repaint

    policy = Gr00tPolicy(..., smooth_option="repaint-euler")
    setup_tensorrt_engines_repaint(policy, "gr00t_engine")

    action = policy.get_action(dict(
        observations=obs,
        inference_delay=2,
        execute_horizon=3,
        prefix_attention_horizon=13,
        actual_action_dim=14,
    ))

To serve it from `scripts/inference_service.py --server --smooth-option repaint-euler
--use-tensorrt`, swap the `setup_tensorrt_engines` import there for
`setup_tensorrt_engines_repaint` — it is a superset (it calls the original, then binds
the repaint entry point on top).

Note on the cold start: `GR00T_N1_5.get_repaint_action` falls back to
`action_head.get_action` on the first chunk, when there is no previous chunk to stay
consistent with. `setup_tensorrt_engines` has already routed that through TensorRT, so
the cold start is accelerated too, with no extra wiring.
"""

import os
from functools import partial

import torch
from transformers.feature_extraction_utils import BatchFeature

from deployment_scripts.trt_model_forward import setup_tensorrt_engines
from gr00t.model.action_head.flow_matching_action_head import get_prefix_weights


def action_head_tensorrt_repaint_forward(
    self,
    action_input: BatchFeature,
    backbone_output: BatchFeature,
    prev_action_chunk: torch.Tensor,  # [B, H, action_dim]
    inference_delay: int,
    prefix_attention_horizon: int,
    actual_action_dim: int,
    use_prev_action: bool = True,
) -> BatchFeature:
    """TensorRT counterpart of `FlowmatchingActionHead.get_repaint_action`.

    Same signature, same five steps, same math — only the denoiser evaluations are
    replaced by engine calls. Everything runs in fp16, which is what the engines are
    built for.
    """
    # ── conditioning (computed once, reused by all three passes) ──────────────────
    # replaces self.process_backbone_output(backbone_output)
    backbone_features = backbone_output.backbone_features
    if backbone_features.dtype != torch.float16:
        backbone_features = backbone_features.to(torch.float16)
    self.vlln_vl_self_attention_engine.set_runtime_tensor_shape(
        "backbone_features", backbone_features.shape
    )
    vl_embs = self.vlln_vl_self_attention_engine(backbone_features)["output"]

    embodiment_id = action_input.embodiment_id
    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)

    # replaces self.state_encoder(action_input.state, embodiment_id)
    state = action_input.state
    if state.dtype != torch.float16:
        state = state.to(torch.float16)
    self.state_encoder_engine.set_runtime_tensor_shape("state", state.shape)
    self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    state_features = self.state_encoder_engine(state, embodiment_id)["output"]

    batch_size = vl_embs.shape[0]
    device = vl_embs.device
    dtype = torch.float16
    d = inference_delay
    num_steps = self.num_inference_timesteps
    dt = 1.0 / num_steps

    future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)

    # prefix_mask[b, i, 0] = True iff i < d  ->  broadcastable over [B, H, D]
    prefix_mask = (
        torch.arange(self.config.action_horizon, device=device).unsqueeze(0).unsqueeze(-1) < d
    )  # [1, H, 1]

    prev_action_chunk = torch.as_tensor(prev_action_chunk, device=device, dtype=dtype)

    # ── shared denoising step ─────────────────────────────────────────────────────
    def model_velocity(actions: torch.Tensor, t: int) -> torch.Tensor:
        """One engine pass of the denoiser at step t. Returns predicted velocity."""
        t_cont = t / float(num_steps)
        t_discretized = int(t_cont * self.num_timestep_buckets)
        timesteps_tensor = torch.full((batch_size,), t_discretized, device=device)

        actions = actions.to(torch.float16)
        self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
        self.action_encoder_engine.set_runtime_tensor_shape(
            "timesteps_tensor", timesteps_tensor.shape
        )
        self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        action_features = self.action_encoder_engine(actions, timesteps_tensor, embodiment_id)[
            "output"
        ]

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).to(
            torch.float16
        )

        self.DiT_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        self.DiT_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
        self.DiT_engine.set_runtime_tensor_shape("timesteps_tensor", timesteps_tensor.shape)
        model_output = self.DiT_engine(sa_embs, vl_embs, timesteps_tensor)["output"]

        self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
        self.action_decoder_engine.set_runtime_tensor_shape(
            "embodiment_id", embodiment_id.shape
        )
        pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
        return pred[:, -self.action_horizon :]

    # ── Step 1: naive forward pass ────────────────────────────────────────────────
    # This attribute is used to ensure the same noise is used for both PyTorch and
    # TensorRT inference, so the two can be compared numerically.
    if hasattr(self, "init_actions"):
        x_0_free = self.init_actions.expand((batch_size, -1, -1)).to(dtype)
    else:
        x_0_free = torch.randn(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            dtype=dtype,
            device=device,
        )

    x = x_0_free.clone()
    for t in range(num_steps):
        x = x + dt * model_velocity(x, t)
    x_1_naive = x

    # ── Step 2: construct inversion target ────────────────────────────────────────
    x_1_target = torch.where(prefix_mask, prev_action_chunk, x_1_naive)

    # ── Step 3: backward Euler inversion ──────────────────────────────────────────
    x = x_1_target.clone().to(dtype=dtype)
    for t in reversed(range(num_steps)):
        x = x - dt * model_velocity(x, t)
    x_0_star = x

    # ── Step 4: Mao re-painting ───────────────────────────────────────────────────
    x_0_repaint = torch.where(prefix_mask, x_0_star, x_0_free)

    # ── Step 5: final forward pass ────────────────────────────────────────────────
    x = x_0_repaint.clone().to(dtype=dtype)
    for t in range(num_steps):
        x = x + dt * model_velocity(x, t)
    x_1_final = x

    print(
        "EACH DENOISING STEP: ",
        (
            x_1_final[:, :inference_delay, :actual_action_dim]
            - prev_action_chunk[:, :inference_delay, :actual_action_dim]
        )
        .abs()
        .mean(),
    )
    if use_prev_action:
        weights = get_prefix_weights(
            inference_delay, prefix_attention_horizon, self.config.action_horizon, "exp"
        ).to(device=device, dtype=dtype)
        x_1_final = prev_action_chunk * weights[:, None] + x_1_final * (1 - weights[:, None])
        print(
            "AFTER ASSIGN: ",
            (x_1_final[:, :inference_delay, :] - prev_action_chunk[:, :inference_delay, :])
            .abs()
            .mean(),
        )

    return BatchFeature(data={"action_pred": x_1_final})


def setup_tensorrt_engines_repaint(
    policy,
    trt_engine_path,
    vit_dtype="fp8",
    llm_dtype="nvfp4",
    dit_dtype="fp8",
):
    """Set up the TensorRT engines and route repaint-euler through them.

    A superset of `setup_tensorrt_engines`: it does everything that does (loads the
    engines, deletes the PyTorch modules they replace, binds the backbone and the
    standard `get_action`), then binds the repaint entry point as well. Drop-in
    replacement wherever `setup_tensorrt_engines` is called today.

    Args:
        policy: GR00T policy model instance, constructed with
            `smooth_option="repaint-euler"`.
        trt_engine_path: Path to the directory containing TensorRT engine files.
        vit_dtype: ViT model dtype (fp16, fp8).
        llm_dtype: LLM model dtype (fp16, nvfp4).
        dit_dtype: DiT model dtype (fp16, fp8).
    """
    if not os.path.isdir(trt_engine_path):
        raise FileNotFoundError(
            f"TensorRT engine directory not found: {trt_engine_path}. "
            f"Build the engines first — see deployment_scripts/README.md."
        )

    setup_tensorrt_engines(policy, trt_engine_path, vit_dtype, llm_dtype, dit_dtype)

    policy.model.action_head.get_repaint_action = partial(
        action_head_tensorrt_repaint_forward, policy.model.action_head
    )
    return policy
