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

from typing import Any, Dict
import torch
import numpy as np
from lerobot.eval.service import BaseInferenceClient, BaseInferenceServer


class PolicyWrapper:
    """Wrapper to make policy compatible with the service interface"""
    
    def __init__(self, policy):
        self.policy = policy
        
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Convert observations to tensors, call policy, and return action as dict"""
        # Convert numpy arrays to tensors and ensure proper device placement
        batch = {}
        for key, value in observations.items():
            if isinstance(value, np.ndarray):
                batch[key] = torch.from_numpy(value).to(device=self.policy.config.device)
            elif isinstance(value, (list, tuple)) and key == "task":
                # Task is typically a list of strings, keep as is
                batch[key] = value
            else:
                batch[key] = value
        
        # Call the policy's select_action method
        action = self.policy.select_action(batch)
        
        # Return as dict format expected by the service
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, np.ndarray):
            pass  # Already numpy
        else:
            action = np.array(action)
            
        return {
            "action": action,
            "success": True
        }

    def reset(self):
        self.policy.reset()


class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "*", port: int = 5555, api_token: str = None):
        super().__init__(host, port, api_token)
        self.wrapped_model = PolicyWrapper(model)
        self.register_endpoint("get_action", self.wrapped_model.get_action)

        self.register_endpoint("reset", self.wrapped_model.reset, requires_input=False)

    @staticmethod
    def start_server(policy, port: int, api_token: str = None):
        server = RobotInferenceServer(policy, port=port, api_token=api_token)
        server.run()


class RobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None):
        super().__init__(host=host, port=port, api_token=api_token)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def reset(self):
        return self.call_endpoint("reset", requires_input=False)