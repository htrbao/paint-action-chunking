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

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.rtc_utils import plot_trajectory

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        smooth_option: Optional[str] = "",
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Gr00tPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path) if (not smooth_option == "training-time-rtc") else self._load_model_action_condition(model_path)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")
            
        self.smooth_option = smooth_option
        if self.smooth_option == "te":
            self.temporal_agg = True
            self.num_queries = 16
            if self.temporal_agg:
                self.k = 0.015
                self.ensemble_weights = torch.exp(-self.k * torch.arange(self.num_queries)).cuda()
                self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0).cuda()
                self.reset()
        elif self.smooth_option == "rtc":
            self.temporal_agg = False
            self.prev_action_chunk = None
            self.cnt = 0

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)
    
    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        self.ensembled_actions_count = None
        self.prev_action_chunk = None
        self.cnt = 0

    def process_output(self, actions):
        """
        Takes a (batch, num_queries, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        # actions = actions * self.stats["action_std"] + self.stats["action_mean"]
        if not self.temporal_agg:
            return actions.squeeze()

        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.num_queries, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, num_queries - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.num_queries)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, :1, :],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        print("PROCESS OUTPUT", action.shape)
        return action

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        if self.smooth_option in ["rtc", "training-time-rtc", "repaint-euler"]:
            inference_delay = obs_copy.get("inference_delay", None)
            prefix_attention_horizon = obs_copy.get("prefix_attention_horizon", None)
            prefix_attention_schedule = obs_copy.get("prefix_attention_schedule", None)
            max_guidance_weight = obs_copy.get("max_guidance_weight", 5)
            sigma_d_o = obs_copy.get("sigma_d_o", 5)
            execute_horizon = obs_copy.get("execute_horizon", None)
            actual_action_dim = obs_copy.get("actual_action_dim", None)
            obs_copy = obs_copy.get("observations", None)

            saved_prev_action_chunk = self.prev_action_chunk
            if self.prev_action_chunk is not None:
                self.prev_action_chunk = torch.concat(
                    (self.prev_action_chunk[:, execute_horizon:],
                    torch.zeros(
                        [self.prev_action_chunk.shape[0], execute_horizon, self.prev_action_chunk.shape[-1]],
                        device=self.prev_action_chunk.device,
                    )),
                    dim=1,
                )

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_action = self._get_action_from_normalized_input(normalized_input) if self.smooth_option not in ["rtc", "training-time-rtc", "repaint-euler"] else self._get_realtime_action_from_normalized_input(
            normalized_input,
            self.prev_action_chunk,
            inference_delay,
            prefix_attention_horizon,
            prefix_attention_schedule,
            max_guidance_weight,
            sigma_d_o,
            actual_action_dim
        )
        if self.smooth_option in ["rtc", "training-time-rtc", "repaint-euler"]:
            normalized_action, self.prev_action_chunk = normalized_action        
            self.cnt += 1
            plot_trajectory(
                {
                    "prev_pred_action_across_time": saved_prev_action_chunk,
                    "pred_action_across_time": normalized_action,
                    "action_dim": actual_action_dim,
                    "executed_horizon": execute_horizon
                },
                save_plot_path=f"eval_{self.smooth_option}_{inference_delay}_{max_guidance_weight}_{sigma_d_o}_{self.cnt}.png"
            )

        unnormalized_action = self._get_unnormalized_action(normalized_action)        

        if self.smooth_option == "te":
            for k in unnormalized_action.keys():
                unnormalized_action[k] = unnormalized_action[k].squeeze(1)  # remove the 1 dimension
        elif not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        if self.smooth_option == "te":
            normalized_action = self.process_output(normalized_action)
        return normalized_action
    
    def _get_realtime_action_from_normalized_input(self, normalized_input: Dict[str, Any],
                                                   prev_action_chunk: torch.Tensor | None,
                                                   inference_delay: int,
                                                   prefix_attention_horizon: int,
                                                   prefix_attention_schedule: str,
                                                   max_guidance_weight: float,
                                                   sigma_d_o: float,
                                                   actual_action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Set up autocast context if needed
        # with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        if self.smooth_option in ["rtc", "training-time-rtc"]:
            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                model_pred, real_action = self.model.get_realtime_action(normalized_input,
                                                                prev_action_chunk=prev_action_chunk,
                                                                inference_delay=inference_delay,
                                                                prefix_attention_horizon=prefix_attention_horizon,
                                                                prefix_attention_schedule=prefix_attention_schedule,
                                                                max_guidance_weight=max_guidance_weight,
                                                                sigma_d_o=sigma_d_o,
                                                                actual_action_dim=actual_action_dim)
        elif self.smooth_option in ["repaint-euler"]:
            model_pred, real_action = self.model.get_repaint_action(normalized_input,
                                                    prev_action_chunk=prev_action_chunk,
                                                    inference_delay=inference_delay,
                                                    prefix_attention_horizon=prefix_attention_horizon,
                                                    actual_action_dim=actual_action_dim)

        normalized_action = model_pred["action_pred"].float()
        real_action = real_action["action_pred"].float()
        return normalized_action, real_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from gr00t.model.action_head.flow_matching_action_head import (
                FlowmatchingActionHead,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHead(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_model_action_condition(self, model_path):
        model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode

        print(
            "Policy: Recreating action head with FlowmatchingActionHeadActionCondition (was FlowmatchingActionHeadActionCondition)"
        )
        
        # Import the FlowmatchingActionHeadActionCondition class
        from gr00t.model.action_head.flow_matching_action_head_action_condition import (
            FlowmatchingActionHeadActionCondition,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHeadActionCondition(model.action_head.config)

        # Copy the weights from the old action head to the new one
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=True)

        # Replace the action head
        model.action_head = new_action_head

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHeadActionCondition class
            from gr00t.model.action_head.flow_matching_action_head_action_condition import (
                FlowmatchingActionHeadActionCondition,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHeadActionCondition(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data
