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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    plot_state=False,
    save_plot_path=None,
    temp_agg=False
):
    if temp_agg:
        action_horizon = 1
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []

    for step_count in range(steps):
        data_point = None
        if plot_state:
            data_point = dataset.get_step_data(traj_id, step_count)
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
            state_joints_across_time.append(concat_state)

        if step_count % action_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data(traj_id, step_count)

            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action(data_point)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

                concat_gt_action = np.concatenate(
                    [data_point[f"action.{key}"][j] for key in modality_keys], axis=0
                )
                gt_action_across_time.append(concat_gt_action)

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    # raise error when pred action has NaN
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Pred action has NaN")

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot or save_plot_path is not None:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "mse": mse,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
        }
        plot_trajectory(info, f"{save_plot_path}.png")

    return mse

def calc_mse_for_overlapping_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    prefix_attention_schedule: str,
    max_guidance_weight: float,
    sigma_d_o: float,
    steps=300,
    action_horizon=16,
    plot=False,
    save_plot_path=None,
):
    policy.reset()
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    pred_action_chunk_across_time = []
    action_chunk = None

    inference_delay = 2
    execute_horizon = 3
    prefix_attention_horizon = action_horizon - execute_horizon


    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point[f"state.{modality_keys[0]}"][0]
        # # concat_gt_action = data_point[f"action.{modality_keys[0]}"][0]
        # concat_state = np.concatenate(
        #     [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        # )
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )

        state_joints_across_time.append(concat_state)
        gt_action_across_time.append(concat_gt_action)

        if step_count % execute_horizon == 0 :
            next_action_chunk = policy.get_action(
                dict(
                    observations=data_point,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    prefix_attention_horizon=prefix_attention_horizon,
                    prefix_attention_schedule=prefix_attention_schedule,
                    max_guidance_weight=max_guidance_weight,
                    sigma_d_o=sigma_d_o,
                    actual_action_dim=14
                )
            )

            if action_chunk is None:
                for j in range(execute_horizon):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(next_action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)
            else:
                for j in range(execute_horizon, execute_horizon + inference_delay, 1):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)
                for j in range(inference_delay, execute_horizon, 1):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(next_action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)

            action_chunk = next_action_chunk

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "mse": mse,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
        }
        plot_trajectory(
            info,
            f"eval_images/eval_rtc/{save_plot_path}_{prefix_attention_schedule}_{traj_id}_{max_guidance_weight}_{sigma_d_o}.png"
        )
        # plt.show()

    return mse

def calc_mse_for_repaint_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    save_plot_path=None,
):
    policy.reset()
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    pred_action_chunk_across_time = []
    action_chunk = None

    inference_delay = 2
    execute_horizon = 3
    prefix_attention_horizon = action_horizon - execute_horizon


    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point[f"state.{modality_keys[0]}"][0]
        # # concat_gt_action = data_point[f"action.{modality_keys[0]}"][0]
        # concat_state = np.concatenate(
        #     [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        # )
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )

        state_joints_across_time.append(concat_state)
        gt_action_across_time.append(concat_gt_action)

        if step_count % execute_horizon == 0 :
            next_action_chunk = policy.get_action(
                dict(
                    observations=data_point,
                    inference_delay=inference_delay,
                    prefix_attention_horizon=prefix_attention_horizon,
                    execute_horizon=execute_horizon,
                    actual_action_dim=14
                )
            )

            if action_chunk is None:
                for j in range(execute_horizon):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(next_action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)
            else:
                for j in range(execute_horizon, execute_horizon + inference_delay, 1):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)
                for j in range(inference_delay, execute_horizon, 1):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(next_action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)

            action_chunk = next_action_chunk

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "mse": mse,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
        }
        plot_trajectory(
            info,
            f"eval_images/eval_repaint/{save_plot_path}_{traj_id}.png"
        )
        # plt.show()

    return mse

def calc_mse_for_action_condition_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    save_plot_path=None,
):
    policy.reset()
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    pred_action_chunk_across_time = []
    action_chunk = None

    inference_delay = 2
    execute_horizon = 3


    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point[f"state.{modality_keys[0]}"][0]
        # # concat_gt_action = data_point[f"action.{modality_keys[0]}"][0]
        # concat_state = np.concatenate(
        #     [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        # )
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )

        state_joints_across_time.append(concat_state)
        gt_action_across_time.append(concat_gt_action)

        if step_count % execute_horizon == 0 :
            next_action_chunk = policy.get_action(
                dict(
                    observations=data_point,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    actual_action_dim=7
                )
            )

            if action_chunk is None:
                for j in range(execute_horizon):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(next_action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)
            else:
                for j in range(execute_horizon, execute_horizon + inference_delay, 1):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)
                for j in range(inference_delay, execute_horizon, 1):
                    # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                    # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                    concat_pred_action = np.concatenate(
                        [np.atleast_1d(next_action_chunk[f"action.{key}"][j]) for key in modality_keys],
                        axis=0,
                    )
                    pred_action_across_time.append(concat_pred_action)

            action_chunk = next_action_chunk

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "mse": mse,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
        }
        plot_trajectory(
            info,
            f"{save_plot_path}_{inference_delay}_{execute_horizon}_{traj_id}.png"
        )
        # plt.show()

    return mse


def plot_trajectory(
    info,
    save_plot_path=None,
):
    """Simple plot of the trajectory with state, gt action, and pred action."""

    # Use non interactive backend for matplotlib if headless
    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = info["action_dim"]
    state_joints_across_time = info["state_joints_across_time"]
    gt_action_across_time = info["gt_action_across_time"]
    pred_action_across_time = info["pred_action_across_time"]
    modality_keys = info["modality_keys"]
    traj_id = info["traj_id"]
    mse = info["mse"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]

    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string[:-2]}\nUnnormalized MSE: {mse:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every ACTION_HORIZON
        for j in range(0, steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
