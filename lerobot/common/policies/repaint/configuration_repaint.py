#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Real Time Chunking (Repaint) and Bidirectional Decoding (BID) configuration classes.

Based on:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
"""

from dataclasses import dataclass

from lerobot.configs.types import RTCAttentionSchedule


@dataclass
class RepaintConfig:
    """Configuration for Real Time Chunking (Repaint) inference.

    Repaint improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = False

    # Core Repaint settings
    # Todo change to exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    execution_horizon: int = 10

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate Repaint configuration parameters."""
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")