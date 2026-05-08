import collections
import dataclasses
import functools
import math
import pathlib
import pickle
from typing import Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import pandas as pd
import tyro

import model as _model
import train_expert


@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    prefix_attention_schedule: _model.PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 5.0


@dataclasses.dataclass(frozen=True)
class FLDMethodConfig:
    """Fast Langevin Dynamics — training-free replacement for gradient-based guidance.
 
    Mirrors the RealtimeMethodConfig signature so it can be swapped in directly.
    The prefix_attention_schedule controls the soft boundary weighting (same as
    realtime_action). The four fld_* parameters control the inner Langevin loop.
    """
    prefix_attention_schedule: _model.PrefixAttentionSchedule = "exp"
    fld_k: int = 5
    fld_lam: float = 8.0
    fld_eta: float = 0.1
    fld_beta: float = 1.0


@dataclasses.dataclass(frozen=True)
class RepaintingMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RepaintingMethodV2Config:
    inversion_method: _model.InversionMethod = "euler"
    optim_iters: int = 5 # only used when inversion_method="optim"
    optim_lr: float = 0.1


@dataclasses.dataclass(frozen=True)
class TEMethodConfig:
    k: float = -0.015
    num_queries: int = 8
    ensemble_weights: jax.Array = dataclasses.field(init=False)
    ensemble_weights_cumsum: jax.Array = dataclasses.field(init=False)

    def __post_init__(self):
        w = jnp.exp(self.k * jnp.arange(self.num_queries))
        object.__setattr__(self, 'ensemble_weights', w)
        object.__setattr__(self, 'ensemble_weights_cumsum', jnp.cumsum(w))


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int | None = None
    num_evals: int = 2048
    num_flow_steps: int = 5

    inference_delay: int = 0
    execute_horizon: int = 1
    method: NaiveMethodConfig | RealtimeMethodConfig | FLDMethodConfig | RepaintingMethodConfig | RepaintingMethodV2Config | TEMethodConfig | BIDMethodConfig = NaiveMethodConfig()

    model: _model.ModelConfig = _model.ModelConfig()


def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model.FlowPolicy,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model.FlowPolicy | None = None,
):
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), config.num_evals
    )
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"

    def execute_chunk(carry, _):
        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, env_state, info)

        rng, obs, env_state, action_chunk, n, te_ensemble, te_counts, te_is_init = carry
        rng, key = jax.random.split(rng)
        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(key, obs, config.num_flow_steps)
        elif isinstance(config.method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            print(
                f"{config.execute_horizon=} {config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            )
            next_action_chunk = policy.realtime_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.prefix_attention_schedule,
                config.method.max_guidance_weight,
            )
        elif isinstance(config.method, FLDMethodConfig):
            # FLD: training-free Langevin replacement for gradient-based guidance.
            # Same prefix/horizon setup as RealtimeMethodConfig; no backprop needed.
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            print(
                f"[FLD] {config.execute_horizon=} {config.inference_delay=} "
                f"{prefix_attention_horizon=} {policy.action_chunk_size=} "
                f"K={config.method.fld_k} lam={config.method.fld_lam} "
                f"eta={config.method.fld_eta} beta={config.method.fld_beta}"
            )
            next_action_chunk = policy.fld_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.prefix_attention_schedule,
                fld_k=config.method.fld_k,
                fld_lam=config.method.fld_lam,
                fld_eta=config.method.fld_eta,
                # fld_beta=config.method.fld_beta,
            )
        elif isinstance(config.method, RepaintingMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            print(
                f"[Repainting-Euler-Origin] {config.execute_horizon=} {config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            )
            next_action_chunk = policy.repainting_action_v2(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay
            )
        elif isinstance(config.method, RepaintingMethodV2Config):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            print(
                f"[RepaintingV2-{config.method.inversion_method}] {config.execute_horizon=} {config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            )
            next_action_chunk = policy.repainting_action_v2(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                config.method.inversion_method
            )
        elif isinstance(config.method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            if config.method.bid_k is not None:
                assert weak_policy is not None, "weak_policy is required for BID"
            next_action_chunk = policy.bid_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
            )
        elif isinstance(config.method, TEMethodConfig):
            next_action_chunk, te_ensemble, te_counts, te_is_init = policy.te_action(
                key, obs, config.num_flow_steps,
                te_ensemble, te_counts, te_is_init,
                config.method.num_queries,
                config.method.ensemble_weights,
                config.method.ensemble_weights_cumsum,
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")

        # we execute `inference_delay` actions from the *previously generated* action chunk, and then the remaining
        # `execute_horizon - inference_delay` actions from the newly generated action chunk
        action_chunk_to_execute = jnp.concatenate(
            [
                action_chunk[:, : config.inference_delay],
                next_action_chunk[:, config.inference_delay : config.execute_horizon],
            ],
            axis=1,
        )
        if config.inference_delay > 0:
            prefix_match = jnp.mean(jnp.abs(next_action_chunk[:, : config.inference_delay] - action_chunk[:, : config.inference_delay]))
        else:
            prefix_match = jnp.array(0.0)
        # throw away the first `execute_horizon` actions from the newly generated action chunk, to align it with the
        # correct frame of reference for the next scan iteration
        next_action_chunk = jnp.concatenate(
            [
                next_action_chunk[:, config.execute_horizon :],
                jnp.zeros((obs.shape[0], config.execute_horizon, policy.action_dim)),
            ],
            axis=1,
        )
        next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
        )
        infos["match"] = prefix_match
        return (rng, next_obs, next_env_state, next_action_chunk, next_n,
                te_ensemble, te_counts, te_is_init), (dones, env_states, infos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)  # [batch, horizon, action_dim]
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
    te_num_queries = config.method.num_queries if isinstance(config.method, TEMethodConfig) else 1
    te_ensemble    = jnp.zeros_like(action_chunk)
    te_counts      = jnp.ones((te_num_queries, 1), dtype=jnp.int32)
    te_is_init     = jnp.array(True)
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n, te_ensemble, te_counts, te_is_init),
        None,
        length=scan_length,
    )
    dones, env_states, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos))
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"
    return_info = {}
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        # only consider the first episode of each rollout
        first_done_idx = jnp.argmax(dones, axis=0)
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video


def main(
    run_path: str,
    config: EvalConfig = EvalConfig(),
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    ),
    seed: int = 0,
    output_dir: str | None = "eval_output",
):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    # load policies from best checkpoints by solve rate
    state_dicts = []
    weak_state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        # load policy
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
        if config.weak_step is not None:
            with (log_dirs[config.weak_step] / "policies" / f"{level_name}.pkl").open("rb") as f:
                weak_state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    if config.weak_step is not None:
        weak_state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *weak_state_dicts))
    else:
        weak_state_dicts = None

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec), out_specs=pspec, check_rep=False)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def _eval(config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict):
        policy = _model.FlowPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.model,
            rngs=nnx.Rngs(rng),
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        if weak_state_dict is not None:
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(weak_state_dict)
            weak_policy = nnx.merge(graphdef, state)
        else:
            weak_policy = None
        eval_info, _ = eval(config, env, rng, level, policy, env_params, static_env_params, weak_policy)
        return eval_info

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    results = collections.defaultdict(list)

    fld_configs = [
        (5,  8.0, 0.1, 1.0, ""),           # default, labelled "fld"
        (2,  8.0, 0.1, 1.0, "_k2"),        # faster, fewer inner steps
        (5, 10.0, 0.1, 1.0, "_lam10"),     # stronger prefix pull
        (5,  8.0, 0.2, 1.0, "_eta02"),     # larger step size
    ]

    for inference_delay in [0, 1, 2, 3, 4]:
        for execute_horizon in range(max(1, inference_delay), 8 - inference_delay + 1):
            print(f"{inference_delay=} {execute_horizon=}")
            for fld_k, fld_lam, fld_eta, fld_beta, suffix in fld_configs:
                c = dataclasses.replace(
                    config, inference_delay=inference_delay, execute_horizon=execute_horizon,
                    method=FLDMethodConfig(
                        fld_k=fld_k,
                        fld_lam=fld_lam,
                        fld_eta=fld_eta,
                        fld_beta=fld_beta,
                    )
                )
                out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append(f"fld{suffix}")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
            
            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon,
                method=RepaintingMethodConfig()
            )
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append(f"repaintv1-euler")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)

            for inv_method in ["rfm", "euler", "dpm2", "optim"]:
                c = dataclasses.replace(
                    config, inference_delay=inference_delay, execute_horizon=execute_horizon,
                    method=RepaintingMethodV2Config(
                        inversion_method=inv_method
                    )
                )
                out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append(f"repaintv2-{inv_method}")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    
            if inference_delay == 0 and execute_horizon == 1:
                c = dataclasses.replace(
                    config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=TEMethodConfig()
                )
                out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append("te")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    
            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig()
            )
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("naive")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)

            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig()
            )
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("realtime")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)

            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig()
            )
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("bid")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)

            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=RealtimeMethodConfig(prefix_attention_schedule="zeros"),
            )
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("hard_masking")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(pathlib.Path(output_dir) / "results.csv", index=False)


if __name__ == "__main__":
    tyro.cli(main)
