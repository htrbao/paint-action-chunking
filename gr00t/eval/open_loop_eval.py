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

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.eval._horizon_contract import PolicyHorizonSpec, migrate_deprecated_action_horizon_argv
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    execution_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        execution_horizon: Number of predicted-chunk steps executed per inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, execution_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def _resolve_action_stride(modality_configs: dict[str, Any]) -> int:
    """Timestep spacing between consecutive entries of the predicted chunk.

    Returns 1 for the usual dense window ``[0, 1, ..., H-1]``. A checkpoint
    trained on a subsampled window such as ``[0, 2, ..., 30]`` returns 2: entry
    ``j`` of the chunk is the action for timestep ``j * stride``, so scoring it
    against a dense ground-truth timeline would compare the wrong rows.

    Only windows that start at 0 and are evenly spaced are supported; anything
    else has no single stride and cannot be aligned this way.
    """
    delta = [int(d) for d in modality_configs["action"].delta_indices]
    if not delta:
        raise ValueError("policy declared an empty action.delta_indices.")
    if len(delta) == 1:
        return 1
    if delta[0] != 0:
        raise ValueError(
            f"action.delta_indices={delta} must start at 0; the eval executes each "
            "chunk starting from the observation timestep."
        )
    strides = {b - a for a, b in zip(delta, delta[1:])}
    if len(strides) != 1 or next(iter(strides)) < 1:
        raise ValueError(
            f"action.delta_indices={delta} is not evenly spaced. The eval aligns "
            "predictions to ground truth by a single stride, which an irregular "
            "window does not have."
        )
    return next(iter(strides))


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    execution_horizon=16,
    save_plot_path=None,
    smooth_option: str | None = None,
    inference_delay: int | None = None,
    async_execution: bool | None = None,
):
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    # Resolve the action window. A checkpoint may predict a temporally
    # subsampled chunk (e.g. delta_indices=[0, 2, ..., 30]: 16 actions at
    # stride 2 spanning 32 timesteps), in which case chunk[j] is the action for
    # timestep j * stride, not j.
    action_stride = _resolve_action_stride(loader.modality_configs)
    if action_stride == 1:
        # Fail fast if the open-loop stride doesn't fit the model's predicted
        # chunk. PolicyHorizonSpec speaks for consumers that index the chunk
        # linearly against a dense timeline, which only holds at stride 1.
        PolicyHorizonSpec.from_modality_config(
            loader.modality_configs, n_action_steps=execution_horizon
        )
    else:
        chunk_len = len(loader.modality_configs["action"].delta_indices)
        if not 1 <= execution_horizon <= chunk_len:
            raise ValueError(
                f"execution_horizon={execution_horizon} must satisfy "
                f"1 <= execution_horizon <= action_horizon={chunk_len}."
            )
        logging.info(
            f"Action chunk is subsampled at stride {action_stride}; evaluating every "
            f"{action_stride}-th timestep, which is where the policy actually predicts."
        )

    # Prefix-consistent chunking carries state across calls, so start each
    # trajectory from a clean slate.
    options = None
    delay = execution_horizon if inference_delay is None else inference_delay
    if smooth_option is not None:
        policy.reset()
        options = {
            "smooth_option": smooth_option,
            "execution_horizon": execution_horizon,
            "inference_delay": delay,
        }

    # Prefix-consistent chunking only shows up at the seam between a chunk still
    # in flight and its replacement, so simulating the delay is what makes the
    # metrics meaningful. Default it on whenever a sampler is selected.
    if async_execution is None:
        async_execution = smooth_option is not None
    if async_execution and not 0 <= delay <= execution_horizon:
        raise ValueError(
            f"inference_delay must be in [0, execution_horizon={execution_horizon}] for "
            f"async execution, got {delay}."
        )

    prev_chunk = None
    prefix_gaps: list[float] = []

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    # Executing `execution_horizon` chunk entries advances the dense timeline by
    # that many *strided* steps.
    for step_count in range(0, actual_steps, execution_horizon * action_stride):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        _action_chunk, _ = policy.get_action(parsed_obs, options)
        action_chunk = parse_action_gr00t(_action_chunk)

        def step_action(chunk, j):
            # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
            return np.concatenate(
                [np.atleast_1d(np.atleast_1d(chunk[f"action.{key}"])[j]) for key in action_keys],
                axis=0,
            )

        if not async_execution or prev_chunk is None:
            for j in range(execution_horizon):
                pred_action_across_time.append(step_action(action_chunk, j))
        else:
            # The robot cannot act on this chunk until it exists, so for
            # `delay` steps it keeps executing the one still in flight, then
            # switches. Those `delay` steps are the committed prefix the new
            # chunk had to agree with.
            chunk_len = len(np.atleast_1d(action_chunk[f"action.{action_keys[0]}"]))
            if execution_horizon + delay > chunk_len:
                raise ValueError(
                    f"async execution needs execution_horizon + inference_delay "
                    f"({execution_horizon} + {delay}) <= predicted chunk length ({chunk_len}); "
                    f"the in-flight chunk runs out of actions to execute during the delay."
                )
            for j in range(execution_horizon, execution_horizon + delay):
                pred_action_across_time.append(step_action(prev_chunk, j))
            for j in range(delay, execution_horizon):
                pred_action_across_time.append(step_action(action_chunk, j))

            # Prefix consistency: how far the new chunk drifts from the actions
            # already committed. This is what PAINT minimizes; MSE against
            # ground truth does not see it.
            if delay > 0:
                prefix_gaps.append(
                    float(
                        np.mean(
                            [
                                np.abs(
                                    step_action(action_chunk, j)
                                    - step_action(prev_chunk, execution_horizon + j)
                                )
                                for j in range(delay)
                            ]
                        )
                    )
                )
        prev_chunk = action_chunk

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints. With a subsampled chunk the policy only predicts every
    # action_stride-th timestep, so ground truth is subsampled to match: entry j
    # of the prediction sequence lines up with dense timestep j * action_stride.
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])[
        :actual_steps:action_stride
    ]
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps:action_stride
    ]
    pred_action_across_time = np.array(pred_action_across_time)
    # The last inference emits a full chunk, which can overrun the trajectory.
    n_eval = min(len(gt_action_across_time), len(pred_action_across_time))
    gt_action_across_time = gt_action_across_time[:n_eval]
    pred_action_across_time = pred_action_across_time[:n_eval]
    state_joints_across_time = state_joints_across_time[:n_eval]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")
    prefix_gap = float(np.mean(prefix_gaps)) if prefix_gaps else None
    if prefix_gap is not None:
        logging.info(
            f"Prefix inconsistency at chunk boundaries (lower is better): {prefix_gap} "
            f"over {len(prefix_gaps)} seams"
        )

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        execution_horizon=execution_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae, prefix_gap


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    execution_horizon: int = 16
    """How many steps of each predicted action chunk to execute before re-planning
    (must be <= the model's predicted chunk length)."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    smooth_option: str | None = None
    """Prefix-consistent chunking mode: 'repaint' (PAINT), 'repaint-euler'
    (alias), or 'rtc' (guidance baseline). None runs the plain sampler."""

    inference_delay: int | None = None
    """Steps assumed already committed when a new chunk lands, for
    --smooth-option. Defaults to --execution-horizon."""

    async_execution: bool | None = None
    """Simulate asynchronous execution: keep executing the in-flight chunk for
    --inference-delay steps before switching to the new one. This is what creates
    the chunk-boundary seam that prefix-consistent sampling targets, and it enables
    the prefix-inconsistency metric. Defaults to on when --smooth-option is set."""


def main(args: ArgsConfig):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Apply --denoising-steps: the action head reads num_inference_timesteps
        # at sampling time.
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        logging.info(f"Using {args.denoising_steps} denoising steps")
    else:
        policy = PolicyClient(host=args.host, port=args.port)
        if args.denoising_steps != ArgsConfig.denoising_steps:
            logging.warning(
                "--denoising-steps=%d is ignored when running against a remote "
                "policy server; set the denoising steps on the server "
                "(run_gr00t_server.py) instead.",
                args.denoising_steps,
            )

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []
    all_prefix_gap = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae, prefix_gap = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            execution_horizon=args.execution_horizon,
            smooth_option=args.smooth_option,
            inference_delay=args.inference_delay,
            async_execution=args.async_execution,
            save_plot_path=args.save_plot_path,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)
        if prefix_gap is not None:
            all_prefix_gap.append(prefix_gap)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
        if all_prefix_gap:
            logging.info(
                f"Average prefix inconsistency across all trajs: "
                f"{np.mean(np.array(all_prefix_gap))}"
            )
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    if migrate_deprecated_action_horizon_argv():
        logging.warning("--action-horizon is deprecated; use --execution-horizon.")
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
