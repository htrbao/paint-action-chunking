import dataclasses
import functools
from typing import Literal, TypeAlias, Self

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


# VLASH dynamics order: which derivatives to include as conditioning
# "position" = action only
# "velocity" = action + velocity (first difference)
# "acceleration" = action + velocity + acceleration (second difference)
VLASHOrder: TypeAlias = Literal["position", "velocity", "acceleration"]


def compute_vlash_state(actions: jax.Array, order: VLASHOrder) -> jax.Array:
    """
    Compute VLASH state from action history for dynamics consistency.
    
    Args:
        actions: Array of shape [..., num_history, action_dim] containing action history
                 where actions[..., 0, :] is the oldest and actions[..., -1, :] is the most recent
        order: Which dynamics order to use
        
    Returns:
        state: Concatenated state vector including position and optionally velocity/acceleration
    """
    if order == "position":
        # Just use the most recent action as position
        return actions[..., -1, :]
    
    elif order == "velocity":
        # Position (most recent action) + velocity (difference between last two)
        position = actions[..., -1, :]
        velocity = actions[..., -1, :] - actions[..., -2, :]
        return jnp.concatenate([position, velocity], axis=-1)
    
    elif order == "acceleration":
        # Position + velocity + acceleration
        position = actions[..., -1, :]
        velocity = actions[..., -1, :] - actions[..., -2, :]
        prev_velocity = actions[..., -2, :] - actions[..., -3, :]
        acceleration = velocity - prev_velocity
        return jnp.concatenate([position, velocity, acceleration], axis=-1)
    
    else:
        raise ValueError(f"Unknown VLASH order: {order}")


def get_vlash_state_dim(action_dim: int, order: VLASHOrder) -> int:
    """Get the dimension of the VLASH state vector."""
    if order == "position":
        return action_dim
    elif order == "velocity":
        return 2 * action_dim
    elif order == "acceleration":
        return 3 * action_dim
    else:
        raise ValueError(f"Unknown VLASH order: {order}")


def get_vlash_history_size(order: VLASHOrder) -> int:
    """Get the required action history size for the given order."""
    if order == "position":
        return 1
    elif order == "velocity":
        return 2
    elif order == "acceleration":
        return 3
    else:
        raise ValueError(f"Unknown VLASH order: {order}")


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8
    simulated_delay: int | None = None
    vlash_order: VLASHOrder = "position"


def posemb_sincos(pos: jax.Array, embedding_dim: int, min_period: float, max_period: float) -> jax.Array:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]
InversionMethod: TypeAlias = Literal["rfm", "dpm2", "euler", "optim"]


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


class MLPMixerBlock(nnx.Module):
    def __init__(
        self, token_dim: int, token_hidden_dim: int, channel_dim: int, channel_hidden_dim: int, *, rngs: nnx.Rngs
    ):
        self.token_mix_in = nnx.Linear(token_dim, token_hidden_dim, use_bias=False, rngs=rngs)
        self.token_mix_out = nnx.Linear(token_hidden_dim, token_dim, use_bias=False, rngs=rngs)
        self.channel_mix_in = nnx.Linear(channel_dim, channel_hidden_dim, use_bias=False, rngs=rngs)
        self.channel_mix_out = nnx.Linear(channel_hidden_dim, channel_dim, use_bias=False, rngs=rngs)
        self.norm_1 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.adaln_1 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        self.adaln_2 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, x: jax.Array, adaln_cond: jax.Array) -> jax.Array:
        scale_1, shift_1, gate_1 = jnp.split(self.adaln_1(adaln_cond), 3, axis=-1)
        scale_2, shift_2, gate_2 = jnp.split(self.adaln_2(adaln_cond), 3, axis=-1)

        # token mix
        residual = x
        x = self.norm_1(x) * (1 + scale_1) + shift_1
        x = x.transpose(0, 2, 1)
        x = self.token_mix_in(x)
        x = nnx.gelu(x)
        x = self.token_mix_out(x)
        x = x.transpose(0, 2, 1)
        x = residual + gate_1 * x

        # channel mix
        residual = x
        x = self.norm_2(x) * (1 + scale_2) + shift_2
        x = self.channel_mix_in(x)
        x = nnx.gelu(x)
        x = self.channel_mix_out(x)
        x = residual + gate_2 * x
        return x


class FlowPolicy(nnx.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.channel_dim = config.channel_dim
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size
        self.simulated_delay = config.simulated_delay
        self.vlash_order = config.vlash_order
        self.ensembled_actions = None
        self.ensembled_actions_count = None

        self.in_proj = nnx.Linear(action_dim + obs_dim, config.channel_dim, rngs=rngs)
        self.mlp_stack = [
            MLPMixerBlock(
                config.action_chunk_size,
                config.token_hidden_dim,
                config.channel_dim,
                config.channel_hidden_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        self.time_mlp = nnx.Sequential(
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
        )
        self.final_norm = nnx.LayerNorm(config.channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.final_adaln = nnx.Linear(
            config.channel_dim, 2 * config.channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.out_proj = nnx.Linear(config.channel_dim, action_dim, rngs=rngs)

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def __call__(self, obs: jax.Array, x_t: jax.Array, time: jax.Array) -> jax.Array:
        assert x_t.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_t.shape
        if time.ndim == 1:
            time = time[:, None]
        time = jnp.broadcast_to(time, (obs.shape[0], self.action_chunk_size))
        time_emb = jax.vmap(
            functools.partial(posemb_sincos, embedding_dim=self.channel_dim, min_period=4e-3, max_period=4.0)
        )(time)
        time_emb = self.time_mlp(time_emb)
        obs = einops.repeat(obs, "b e -> b c e", c=self.action_chunk_size)
        x = jnp.concatenate([x_t, obs], axis=-1)
        x = self.in_proj(x)
        for mlp in self.mlp_stack:
            x = mlp(x, time_emb)
        assert x.shape == (obs.shape[0], self.action_chunk_size, self.channel_dim), x.shape
        scale, shift = jnp.split(self.final_adaln(time_emb), 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.out_proj(x)
        return x

    def action(self, rng: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1

    def bid_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        n_samples: int,
        # when below two are None, it becomes backwards loss only (i.e., rejection sampling)
        bid_weak_policy: Self | None = None,
        bid_k: int | None = None,
    ) -> jax.Array:
        obs = einops.repeat(obs, "b ... -> (n b) ...", n=n_samples)
        weights = get_prefix_weights(inference_delay, prefix_attention_horizon, self.action_chunk_size, "exp")

        def backward_loss(action_chunks: jax.Array):
            error = jnp.linalg.norm(action_chunks - prev_action_chunk, axis=-1)  # [n, b, h]
            return jnp.sum(error * weights[None, None, :], axis=-1)  # [n, b]

        strong_actions = einops.rearrange(self.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples)
        loss = backward_loss(strong_actions)  # [n, b]

        if bid_weak_policy is not None or bid_k is not None:
            assert bid_weak_policy is not None and bid_k is not None, (bid_weak_policy, bid_k)
            weak_actions = einops.rearrange(
                bid_weak_policy.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples
            )
            weak_loss = backward_loss(weak_actions)  # [n, b]
            weak_idxs = jax.lax.top_k(-weak_loss.T, bid_k)[1].T  # [k, b]
            strong_idxs = jax.lax.top_k(-loss.T, bid_k)[1].T  # [k, b]
            a_plus = jnp.take_along_axis(strong_actions, strong_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            a_minus = jnp.take_along_axis(weak_actions, weak_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            # compute forward loss for each action in strong_actions
            forward_loss = jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_plus[None, :], axis=-1),  # [n, k, b, h]
                axis=(1, 3),  # [n, b]
            ) - jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_minus[None, :], axis=-1),  # [n, k, b, h]
                axis=(1, 3),  # [n, b]
            )
            loss += forward_loss / n_samples

        best_idxs = jnp.argmin(loss, axis=0)  # [b]
        return jnp.take_along_axis(strong_actions, best_idxs[None, :, None, None], axis=0).squeeze(0)  # [b, h, d]

    def realtime_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule,
        max_guidance_weight: float,
    ) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch
            def pinv_corrected_velocity(obs, x_t, y, t):
                def denoiser(x_t):
                    v_t = self(obs[None], x_t[None], t)[0]
                    return x_t + v_t * (1 - t), v_t

                x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay, prefix_attention_horizon, self.action_chunk_size, prefix_attention_schedule
                )
                error = (y - x_1) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]
                # constants from paper
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
                c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                return v_t + guidance_weight * pinv_correction

            if self.simulated_delay is not None:
                mask = jnp.arange(self.action_chunk_size)[None, :] < inference_delay
                x_t = jnp.where(mask[:, :, None], prev_action_chunk, x_t)
                time_chunk = jnp.where(mask, 1.0, time)
                v_t = self(obs, x_t, time_chunk)
            else:
                v_t = pinv_corrected_velocity(obs, x_t, prev_action_chunk, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1
    
    def fld_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,   # [B, H, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule,
        fld_k: int = 1,          # inner refinement sub-steps per outer ODE step
        fld_lam: float = 5.0,    # guidance strength (analogous to max_guidance_weight)
        fld_eta: float = 0.1,    # inner sub-step size  (must satisfy k*eta << dt)
    ) -> jax.Array:
        """Gradient-free velocity-correction inpainting for flow matching ODEs.
 
        Core principle
        ──────────────
        All guidance must live in *velocity space*, never in x_t space.
 
        Wrong (breaks ODE — what the broken LanPaint port did):
            x_t += η · drift + √(2η) · noise
 
        Correct (this method):
            v_t  += drift / (1 − t)          # chain rule through x̂₁ = x_t + (1-t)·v_t
            x_t  += dt · v_corrected         # outer Euler step, ODE manifold preserved
 
        Why: the flow ODE is deterministic. x_t must follow the learned flow manifold.
        Any direct perturbation of x_t is seen by the next denoiser call as an
        out-of-distribution input — especially catastrophic in low-dimensional action
        spaces (6-DOF) where noise magnitude equals signal magnitude.
 
        Algorithm per outer ODE step
        ────────────────────────────
        1. Compute adaptive lam_t  (strong near t=0, fades toward t=1)
        2. Inner loop (K times):
               a. forward pass  →  v_t = model(obs, x_t, time)
               b. predict clean: x̂₁ = x_t + (1-t) · v_t
               c. drift in x̂₁-space: δ = lam_t · (prev_chunk − x̂₁) · weights
               d. velocity correction: Δv = δ / (1-t)        [chain rule]
               e. sub-step along corrected v — NO noise:
                      x_t ← x_t + eta · (v_t + Δv)
        3. Final forward pass at refined x_t
        4. Apply velocity correction once more and commit Euler step:
               x_t ← x_t + dt · (v_t_final + Δv_final)
 
        lam_t schedule
        ──────────────
        lam_t = fld_lam · (1 − time)
 
        This has a critical numerical property: when substituted into Δv,
            Δv = [fld_lam · (1−t) · (prev − x̂₁) · w] / (1−t)
               = fld_lam · (prev − x̂₁) · w
 
        The (1−t) cancels exactly, giving a constant-magnitude correction
        throughout the ODE with no division-by-zero risk at t → 1.
 
        Guidance is naturally stronger at the start (t=0, pure noise, large
        influence over final output) and drops to zero at the end (t=1, clean).
        This mirrors realtime_action's guidance_weight schedule.
 
        Inner sub-steps (fld_k)
        ───────────────────────
        Each sub-step is one gradient-free Newton step on the constraint
        ||prev_chunk − x̂₁(x_t)||². With the (1−t) cancellation above,
        the effective step is:
            x_t ← x_t + eta · fld_lam · (prev − x̂₁) · w  +  eta · v_t
 
        Convergence requires: eta · fld_lam < 1.
        Compute cost: (fld_k + 1) · num_steps denoiser calls total.
        fld_k=1 (default) is 2× naive — the minimal useful setting.
        fld_k=0 skips the inner loop; only the final corrected step applies.
 
        Args:
            rng:                       JAX random key (kept for API compatibility,
                                       not used — this method is deterministic)
            obs:                       Observations  [B, obs_dim]
            num_steps:                 Number of outer ODE steps
            prev_action_chunk:         Previously executed chunk  [B, H, action_dim]
            inference_delay:           Number of prefix actions that are "known"
            prefix_attention_horizon:  Position at which prefix weight decays to zero
            prefix_attention_schedule: Decay curve shape (e.g. "exp", "linear")
            fld_k:    Inner refinement sub-steps per outer step.
                      0 = single corrected forward pass (cheapest, still effective).
                      1 = one refinement then corrected commit (recommended).
                      >1 = more refinement, diminishing returns beyond 3.
            fld_lam:  Guidance strength. Analogous to max_guidance_weight.
                      With the schedule above the effective correction magnitude is
                      fld_lam · ||prev − x̂₁|| · weight, constant in t.
                      Typical range: 3–8. Start at 5 and tune.
            fld_eta:  Inner sub-step size.  Must satisfy: fld_k · fld_eta << dt.
                      With num_steps=5, dt=0.2.  fld_k=1, eta=0.1 → 0.1 << 0.2 ✓.
 
        Returns:
            x_1: Generated action chunk  [B, action_chunk_size, action_dim]
        """
        dt = 1.0 / num_steps
 
        # weights[h] ∈ [0, 1]: guidance strength at action index h.
        # 1.0 on the prefix (h < inference_delay), decaying to 0 beyond
        # prefix_attention_horizon, 0 on the free region beyond that.
        weights = get_prefix_weights(
            inference_delay,
            prefix_attention_horizon,
            self.action_chunk_size,
            prefix_attention_schedule,
        )  # [H]
 
        def _corrected_velocity(x_t, v_t, time, lam_t):
            """Return v_t + Δv, where Δv steers x̂₁ toward prev_action_chunk.
 
            x_t is read-only here.  The returned value is a corrected velocity
            estimate — drop-in replacement for v_t in any Euler step.
 
            With lam_t = fld_lam*(1-t):
                Δv = lam_t*(prev - x̂₁)*w / (1-t)
                   = fld_lam*(prev - x̂₁)*w      ← (1-t) cancels, always finite
            """
            x1_pred     = x_t + (1.0 - time) * v_t                # [B, H, D]
            drift       = lam_t * (prev_action_chunk - x1_pred) \
                          * weights[None, :, None]                  # [B, H, D]
            v_correction = drift / (1.0 - time + 1e-6)             # [B, H, D]
            return v_t + v_correction
 
        def step(carry, _):
            x_t, time = carry
 
            # Adaptive guidance: lam_t · (1-t) schedule so that the (1-t)
            # factor in Δv = drift/(1-t) cancels → bounded correction for all t.
            # Strong early (shapes trajectory), zero at t=1 (clean output).
            lam_t = fld_lam * (1.0 - time)
 
            # ── Inner refinement: move x_t toward corrected ODE path ──────────
            # Each sub-step is one forward pass + corrected Euler micro-step.
            # NO noise — x_t must stay on the flow manifold.
            def inner_step(x_t_inner, _):
                v_t_inner  = self(obs, x_t_inner, time)
                v_corrected = _corrected_velocity(x_t_inner, v_t_inner, time, lam_t)
                return x_t_inner + fld_eta * v_corrected, None
 
            x_t, _ = jax.lax.scan(inner_step, x_t, None, length=fld_k)
 
            # ── Commit outer ODE step with velocity correction ────────────────
            # Re-evaluate v_t at the refined x_t, apply correction, advance.
            # This ensures the full outer Euler step respects the prefix.
            v_t_final   = self(obs, x_t, time)
            v_committed = _corrected_velocity(x_t, v_t_final, time, lam_t)
 
            return (x_t + dt * v_committed, time + dt), None
 
        x_0 = jax.random.normal(
            rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim)
        )
        (x_1, _), _ = jax.lax.scan(step, (x_0, 0.0), None, length=num_steps)
 
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim)
        return x_1

    def repainting_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,   # [B, H, action_dim]
        inference_delay: int
    ) -> jax.Array:
        """
        Real-time action generation via DDIM-inversion re-painting.

        Adapts the re-painting idea from Mao et al. "Guided Image Synthesis via
        Initial Image Editing in Diffusion Model" (MM'23) to flow matching action
        chunks.

        Core insight (Mao et al.)
        """
        dt = 1.0 / num_steps
 
        def forward_step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t + dt * v_t, time + dt), None
 
        def backward_step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t - dt * v_t, time - dt), None
 
        # prefix_mask: True for positions [:d], False for [d:]
        prefix_mask = (
            jnp.arange(self.action_chunk_size)[None, :, None] < inference_delay
        )  # [1, H, 1]
 
        # ── Step 1: forward — get on-manifold x_1_naive for step 2 ───────────
        rng, rng_free, rng_fresh = jax.random.split(rng, 3)
        x_0_free = jax.random.normal(
            rng_free, shape=(obs.shape[0], self.action_chunk_size, self.action_dim)
        )
        (x_1_naive, _), _ = jax.lax.scan(forward_step, (x_0_free, 0.0), length=num_steps)
 
        # ── Step 2: construct target ───────────────────────────────────────────
        x_1_target = jnp.where(prefix_mask, prev_action_chunk, x_1_naive)
        # [:d] = prev_action_chunk[:d]   hard-set prefix
        # [d:] = x_1_naive[d:]           on-manifold free (needed for valid inversion)
 
        # # ── Step 3: backward — invert to find x_0_star ──────────────────────── THIS IS EULER BACKWARD
        (x_0_star, _), _ = jax.lax.scan(
            backward_step, (x_1_target, 1.0 - dt), length=num_steps
        )
        # # x_0_star[:d] ← the noise that denoises to prev_action_chunk[:d]  KEEP THIS
        # # x_0_star[d:] ← the noise that denoises to x_1_naive[d:]          DISCARD
 
        # ── Step 4: Mao re-painting — KEEP prefix noise, REPLACE free noise ──
        fresh_eps = jax.random.normal(
            rng_fresh, shape=(obs.shape[0], self.action_chunk_size, self.action_dim)
        )
        x_0_repaint = jnp.where(prefix_mask, x_0_star, x_0_free)
        # [:d] = x_0_star[:d]   KEPT  — anchors output to prev_action_chunk[:d]
        # [d:] = fresh_eps[d:]  REPLACED — model repaints from fresh Gaussian noise
 
        # ── Step 5: forward from re-painted noise ─────────────────────────────
        (x_1_final, _), _ = jax.lax.scan(forward_step, (x_0_repaint, 0.0), length=num_steps)
        # x_1_final[:d] ≈ prev_action_chunk[:d]
        # x_1_final[d:] ~ fresh model generation from obs
 
        assert x_1_final.shape == (obs.shape[0], self.action_chunk_size, self.action_dim)
        return x_1_final

    def repainting_action_v2(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,   # [B, H, action_dim]
        inference_delay: int,
        inversion_method: str = "rfm",  # "rfm" | "dpm2" | "euler" | "optim"
        optim_iters: int = 5,           # only used when inversion_method="optim"
        optim_lr: float = 0.1,
    ) -> jax.Array:
        """
        Repainting with swappable inversion method.
 
        Compares all inversion approaches in step 3.
        Steps 1, 2, 4, 5 are identical regardless of inversion method.
 
        Method costs (total model calls):
            rfm:   N + 1 + N = 2N + 1   (cheapest, best quality/cost)
            dpm2:  N + 2N + N = 4N      (better accuracy, 2× Euler cost)
            euler: N + N + N  = 3N      (baseline)
            optim: N + k·N + N = (k+2)N (most accurate, expensive)
        """
        dt = 1.0 / num_steps
 
        prefix_mask = (
            jnp.arange(self.action_chunk_size)[None, :, None] < inference_delay
        )  # [1, H, 1]
 
        def forward_scan(x_0_init):
            def step(carry, _):
                x_t, time = carry
                v_t = self(obs, x_t, time)
                return (x_t + dt * v_t, time + dt), None
            (x_1, _), _ = jax.lax.scan(step, (x_0_init, 0.0), length=num_steps)
            return x_1
 
        # ── Steps 1 & 2: forward + construct target ───────────────────────────
        rng, rng_free = jax.random.split(rng)
        x_0_free  = jax.random.normal(
            rng_free, shape=(obs.shape[0], self.action_chunk_size, self.action_dim)
        )
        x_1_naive  = forward_scan(x_0_free)
        x_1_target = jnp.where(prefix_mask, prev_action_chunk, x_1_naive)
 
        # ── Step 3: inversion — swappable ─────────────────────────────────────
        if inversion_method == "rfm":
            x_0_star = self._invert_rfm(obs, x_1_target, num_steps)
 
        elif inversion_method == "dpm2":
            x_0_star = self._invert_dpm2(obs, x_1_target, num_steps)
 
        elif inversion_method == "euler":
            x_0_star = self._invert_euler(obs, x_1_target, num_steps)
 
        elif inversion_method == "optim":
            x_0_star = self._invert_optimization(
                obs, x_1_target, num_steps,
                num_iters=optim_iters, lr=optim_lr,
                init_method="zeros",   # warm-start from RFM
            )
        else:
            raise ValueError(f"Unknown inversion_method: {inversion_method}")
 
        # ── Steps 4 & 5: Mao re-painting + forward ────────────────────────────
        x_0_repaint = jnp.where(prefix_mask, x_0_star, x_0_free)
        x_1_final   = forward_scan(x_0_repaint)
 
        assert x_1_final.shape == (obs.shape[0], self.action_chunk_size, self.action_dim)
        return x_1_final

    def te_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        te_ensemble: jax.Array,          # [B, H, action_dim]  — from carry
        te_counts: jax.Array,            # [num_queries, 1]    — from carry
        te_is_init: jax.Array,           # scalar bool          — from carry
        num_queries: int,
        ensemble_weights: jax.Array,     # [num_queries]
        ensemble_weights_cumsum: jax.Array,  # [num_queries]
    ):
        """
        Temporal Ensembling (TE) — stateless JAX-scan-compatible version.
 
        Implements the exponential weighting scheme of Zhao et al. [2023]:
            w_i = exp(-m * i),   where i is the age of the prediction
 
        State is passed through the scan carry rather than stored on self,
        so jax.lax.cond can branch correctly at runtime.
 
        Returns
        -------
        action_chunk   : [B, H, action_dim]  ensemble output (full chunk)
        te_ensemble    : [B, H, action_dim]  updated ensemble for next carry
        te_counts      : [num_queries, 1]    updated counts for next carry
        te_is_init     : scalar bool         always False after first call
        """
        x_1 = self.action(rng, obs, num_steps)
        B = x_1.shape[0]
 
        # ── INIT: first call — no prior ensemble to blend with ────────────────
        def init_branch(_):
            new_ensemble = x_1                                         # [B, H, D]
            new_counts   = jnp.ones((num_queries, 1), dtype=jnp.int32)
            return new_ensemble, new_counts
 
        # ── UPDATE: blend new chunk with existing ensemble ────────────────────
        # For positions 0..H-2: weighted average between old and new prediction
        # For position H-1: always take the newest prediction (no prior overlap)
        def update_branch(_):
            counts = te_counts                                         # [num_queries, 1]
 
            w_prev = ensemble_weights_cumsum[counts - 1]               # [num_queries, 1]
            w_new  = ensemble_weights[counts]                          # [num_queries, 1]
            w_sum  = ensemble_weights_cumsum[counts]                   # [num_queries, 1]
 
            # blend overlapping positions  [B, H-1, D]
            blended = (
                te_ensemble[:, :-1, :] * w_prev
                + x_1[:, :-1, :] * w_new
            ) / w_sum
 
            new_ensemble = jnp.concatenate(
                [blended, x_1[:, -1:, :]], axis=1
            )                                                          # [B, H, D]
 
            new_counts = jnp.concatenate([
                jnp.clip(counts + 1, a_max=num_queries),               # [num_queries, 1]
                jnp.ones((1, 1), dtype=counts.dtype),                  # new tail slot
            ], axis=0)                                                 # [num_queries+1, 1]
 
            # drop the oldest count slot to keep shape [num_queries, 1]
            new_counts = new_counts[:num_queries]
 
            return new_ensemble, new_counts
 
        # ── runtime branch via jax.lax.cond ──────────────────────────────────
        new_ensemble, new_counts = jax.lax.cond(
            te_is_init,
            init_branch,
            update_branch,
            None,
        )
 
        # ── slide: remove first action, pad end with zeros ───────────────────
        # eval_flow.py will further slice [inference_delay:execute_horizon]
        # from the returned chunk; we slide the internal buffer here to align
        # the ensemble window with the next execute_chunk call.
        next_ensemble = jnp.concatenate([
            new_ensemble[:, 1:, :],
            jnp.zeros((B, 1, self.action_dim), dtype=new_ensemble.dtype),
        ], axis=1)                                                     # [B, H, D]
 
        next_counts = jnp.concatenate([
            new_counts[1:],                                            # drop oldest
            jnp.ones((1, 1), dtype=new_counts.dtype),                 # new slot
        ], axis=0)                                                     # [num_queries, 1]
 
        return new_ensemble, next_ensemble, next_counts, jnp.array(False)
    
    def vlash_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        action_history: jax.Array,  # [batch, history_size, action_dim] for full-order dynamics
    ) -> jax.Array:
        """
        Generate action chunk with VLASH conditioning.
        
        Args:
            rng: Random key
            obs: Observation [batch, obs_dim]
            num_steps: Number of flow steps
            action_history: Action history [batch, history_size, action_dim]
                           history_size should be >= get_vlash_history_size(self.vlash_order)
                           
        Returns:
            action_chunk: [batch, action_chunk_size, action_dim]
        """
        # Compute state from action history based on dynamics order
        state = compute_vlash_state(action_history, self.vlash_order)
        obs = jnp.concatenate([obs, state], axis=-1)
        return self.action(rng, obs, num_steps)

    def loss(self, rng: jax.Array, obs: jax.Array, action: jax.Array):
        assert action.dtype == jnp.float32
        assert action.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), action.shape
        noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
        time = jax.random.uniform(time_rng, (obs.shape[0],))
        noise = jax.random.normal(noise_rng, shape=action.shape)
        u_t = action - noise

        if self.simulated_delay is None:
            x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * action
            pred = self(obs, x_t, time)
            return jnp.mean(jnp.square(pred - u_t))

        w = jnp.exp(jnp.arange(0, self.simulated_delay)[::-1])
        w = w / jnp.sum(w)
        delay = jax.random.choice(delay_rng, self.simulated_delay, (obs.shape[0],), p=w)
        mask = jnp.arange(self.action_chunk_size)[None, :] < delay[:, None]
        time = jnp.where(mask, 1.0, time[:, None])
        x_t = (1 - time[:, :, None]) * noise + time[:, :, None] * action
        pred = self(obs, x_t, time)
        loss = jnp.square(pred - u_t)
        loss_mask = jnp.logical_not(mask)[:, :, None]
        return jnp.sum(loss * loss_mask) / (jnp.sum(loss_mask) + 1e-8)

    def forward_shared_observation(
        self,
        rng: jax.Array,
        obs: jax.Array,                # [B, N, obs_dim] - observation for each offset
        states: jax.Array,             # [B, N, vlash_state_dim] - computed from action history
        actions: jax.Array,            # [B, N, H, action_dim]
    ):
        """
        Shared-observation training forward pass for VLASH.
        
        For each sample in the batch, we have N different simulated delay offsets.
        Each offset corresponds to a different observation point where we compute the
        VLASH state from action history and predict action chunks.
        
        Args:
            rng: Random key
            obs: Observations for each offset [B, N, obs_dim]
            states: VLASH states for each offset, computed from action history [B, N, vlash_state_dim]
                    vlash_state_dim depends on vlash_order:
                    - "position": action_dim
                    - "velocity": 2 * action_dim  
                    - "acceleration": 3 * action_dim
            actions: Target action chunks for each offset [B, N, H, action_dim]
        """
        assert actions.dtype == jnp.float32
        assert actions.shape == (obs.shape[0], states.shape[1], self.action_chunk_size, self.action_dim), actions.shape
        assert obs.shape[:2] == states.shape[:2], f"{obs.shape=} {states.shape=}"
        noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
        batch_size, offset, _ = states.shape
        time = jax.random.uniform(time_rng, (batch_size, offset))
        noise = jax.random.normal(noise_rng, shape=actions.shape)
        u_t = actions - noise

        time_exp = time[:, :, None, None]          # [B, N, 1, 1]

        x_t = (1 - time_exp) * noise + time_exp * actions

        # Flatten offsets
        obs_flat = obs.reshape(batch_size * offset, -1)
        states_flat = states.reshape(batch_size * offset, -1)
        x_t_flat = x_t.reshape(batch_size * offset, self.action_chunk_size, -1)
        time_flat = time.reshape(batch_size * offset, -1)

        pred = self(jnp.concatenate([obs_flat, states_flat], axis=-1), x_t_flat, time_flat)

        pred = einops.rearrange(pred, "(b n) c e -> b n c e", b=batch_size)

        return jnp.mean(jnp.square(pred - u_t))
    
    def _invert_rfm(
        self,
        obs: jax.Array,
        x_1_target: jax.Array,
        num_steps: int,          # unused — single model call regardless
    ) -> jax.Array:
        """
        RFM single-step inversion (arxiv:2601.08136 Remark 5).

        Derivation
        ──────────
        Flow matching linear interpolant: x_t = (1-t)·x_0 + t·x_1
        Velocity target:                  v = x_1 - x_0
        Noise prediction at t=1:          x̂_0 = x_1 - v_θ(x_1, t=1)

        This is exact for a perfectly straight flow (OT-FM).
        For a real model it has error proportional to flow curvature
        (how non-straight the learned trajectories are), NOT to dt.

        Cost: 1 model call.
        Error: O(flow curvature) — not O(dt), so increasing N does not help.
        Best when: flow is well-trained with OT objective (straightest trajectories).
        """
        t_one    = jnp.ones((obs.shape[0],))
        v_at_1   = self(obs, x_1_target, t_one)
        x_0_star = x_1_target - v_at_1
        return x_0_star

    def _invert_euler(
        self,
        obs: jax.Array,
        x_1_target: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """
        Naive backward Euler inversion.

        Algorithm per step (t goes from 1-dt down to 0):
            x_{t-dt} = x_t - dt · v_θ(x_t, t)

        Cost: N model calls.
        Error: O(dt) per step = O(1/N) total.
        At N=5, dt=0.2: error O(0.2) per step — significant.
        Increasing N improves quality but adds cost proportionally.

        This is the baseline — other methods should outperform it.
        """
        dt = 1.0 / num_steps

        def backward_step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t - dt * v_t, time - dt), None

        (x_0_star, _), _ = jax.lax.scan(
            backward_step, (x_1_target, 1.0 - dt), length=num_steps
        )
        return x_0_star

    def _invert_dpm2(
        self,
        obs: jax.Array,
        x_1_target: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """
        DPM2 / Heun second-order backward inversion.

        Algorithm per step (backward, t: 1 → 0):
            k1   = v_θ(x_t, t)              — velocity at current point
            x_mid = x_t - (dt/2) · k1       — half-step predictor (backward)
            k2   = v_θ(x_mid, t - dt/2)     — velocity at midpoint
            x_{t-dt} = x_t - dt · k2        — corrected full step

        Why this works
        ──────────────
        Euler uses velocity at the START of each step (t).
        DPM2 uses velocity at the MIDPOINT (t - dt/2).
        The midpoint estimate cancels the first-order error term:

            Error (Euler): O(dt)    at N=5 → O(0.2) per step
            Error (DPM2):  O(dt²)   at N=5 → O(0.04) per step  ←  5× better

        This is equivalent to Runge-Kutta RK2 applied to the backward ODE.

        Cost: 2N model calls (2 per step).
        Error: O(dt²) per step — 4× more accurate than Euler at same N.
        Tradeoff: 2× cost of Euler, same quality as Euler with 2× more steps.

        Best when: N is small (5-10), where Euler error is large but
                    doubling N would be too expensive.
        """
        dt = 1.0 / num_steps

        def backward_step_dpm2(carry, _):
            x_t, time = carry

            # predictor: half backward Euler step
            k1    = self(obs, x_t, time)
            x_mid = x_t - 0.5 * dt * k1

            # corrector: evaluate at midpoint time
            k2    = self(obs, x_mid, time - 0.5 * dt)

            # full step using midpoint velocity
            x_prev = x_t - dt * k2
            return (x_prev, time - dt), None

        (x_0_star, _), _ = jax.lax.scan(
            backward_step_dpm2, (x_1_target, 1.0 - dt), length=num_steps
        )
        return x_0_star

    def _invert_optimization(
        self,
        obs: jax.Array,
        x_1_target: jax.Array,
        num_steps: int,
        num_iters: int = 5,
        lr: float = 0.1,
        init_method: str = "rfm",    # "rfm" | "euler" | "zeros" | "random"
        rng: jax.Array | None = None,
    ) -> jax.Array:
        """
        Optimization-based inversion — most accurate, most expensive.

        Solves:  x_0* = argmin_{x_0} ||forward_ODE(x_0, obs) - x_1_target||²

        This is exact up to optimization convergence — error is not tied to dt
        but to how many gradient steps we take and how well-conditioned the
        forward ODE is as a function of x_0.

        Algorithm
        ──────────
        1. Initialize x_0 (warm-start from RFM or Euler for faster convergence)
        2. For each iteration:
                x_1_hat = forward_ODE(x_0, obs)      [N model calls, differentiable]
                loss    = ||x_1_hat - x_1_target||²
                x_0    -= lr · ∇_{x_0} loss          [backprop through ODE]

        Warm-start initialization
        ──────────────────────────
        Starting from the RFM estimate (1 model call) dramatically reduces
        the number of gradient iterations needed — typically 3-5 iterations
        from RFM init achieves better quality than 20+ from random init.

        Cost: init_cost + num_iters × N model calls (forward) + backprop.
                At num_iters=5, N=5: 25 forward + backprop ≈ 50 effective calls.
        Error: converges to zero with enough iterations (given sufficient lr tuning).
        Warning: jax.grad through jax.lax.scan — memory cost O(N) for gradient tape.

        Best when: quality is critical and compute budget allows it.
                    Use as an upper bound on what inversion can achieve.
        """
        dt = 1.0 / num_steps

        # ── differentiable forward ODE (needed for grad) ──────────────────────
        def forward_ode(x_0):
            def step(carry, _):
                x_t, time = carry
                v_t = self(obs, x_t, time)
                return (x_t + dt * v_t, time + dt), None
            (x_1, _), _ = jax.lax.scan(step, (x_0, 0.0), length=num_steps)
            return x_1

        # ── loss: squared distance to target ─────────────────────────────────
        def loss_fn(x_0):
            x_1_hat = forward_ode(x_0)
            return jnp.mean(jnp.sum((x_1_hat - x_1_target) ** 2, axis=(-2, -1)))

        # ── warm-start initialization ─────────────────────────────────────────
        if init_method == "rfm":
            # best init: RFM single-step (1 call, close to solution)
            x_0 = self._invert_rfm(obs, x_1_target, num_steps)
        elif init_method == "euler":
            # decent init: backward Euler (N calls)
            x_0 = self._invert_euler(obs, x_1_target, num_steps)
        elif init_method == "zeros":
            x_0 = jnp.zeros_like(x_1_target)
        else:
            # random: requires rng
            assert rng is not None, "rng required for random init"
            x_0 = jax.random.normal(rng, shape=x_1_target.shape)

        # ── gradient descent ──────────────────────────────────────────────────
        # use jax.grad for clean gradient computation through the ODE
        grad_fn = jax.grad(loss_fn)

        def opt_step(x_0, _):
            g    = grad_fn(x_0)
            x_0  = x_0 - lr * g
            return x_0, None

        x_0_star, _ = jax.lax.scan(opt_step, x_0, None, length=num_iters)
        return x_0_star
