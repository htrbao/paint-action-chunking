# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PATH"] = "/mnt/lustre-grete/usr/u12045/projects/LLAVA-Med/envs/lerobot/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/mnt/lustre-grete/usr/u12045/vla/hf_cache"
os.environ["TMPDIR"] = "/mnt/lustre-grete/usr/u12045/vla/cache"
os.environ["PYTHONPATH"] = "/mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid:" + os.environ.get("PYTHONPATH", "")
import sys
sys.path.insert(0, "/mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid")


import json
import pickle
from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig


def display(tensor: torch.Tensor):
    if tensor.dtype == torch.bool:
        tensor = tensor.float()
    print(f"Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean().item()}")
    print(f"Std: {tensor.std().item()}")
    print(f"Min: {tensor.min().item()}")
    print(f"Max: {tensor.max().item()}")


def main():
    num_motors = 15
    device = "cuda"
    dataset_repo_id = "ducido/calvin_task_D_D_scale_50_lerobo_format"

    ckpt_torch_dir = Path(f"/mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid/calvin_jaxcp_conversion_to_torch")
    ckpt_jax_dir = Path(f"/mnt/lustre-grete/usr/u12045/vla/duci/openpi/checkpoints/pi0_calvin_50%_joint/pi0_calvin_50%_joint/30000")
    save_dir = Path(f"/mnt/lustre-grete/usr/u12045/vla/duci/openpi/example_sample")

    with open(save_dir / "example.pkl", "rb") as f:
        example = pickle.load(f)
    with open(save_dir / "outputs.pkl", "rb") as f:
        outputs = pickle.load(f)
    with open(save_dir / "noise.pkl", "rb") as f:
        noise = pickle.load(f)
    with open(save_dir / "inputs_transform.pkl", "rb") as f:
        jax_batch_norm = pickle.load(f)

    with open(ckpt_jax_dir / f"assets/{dataset_repo_id}/norm_stats.json") as f:
        norm_stats = json.load(f)

    # Override stats
    dataset_meta = LeRobotDatasetMetadata(dataset_repo_id)
    dataset_meta.stats["state"]["mean"] = torch.tensor(
        norm_stats["norm_stats"]["state"]["mean"][:num_motors], dtype=torch.float32
    )
    dataset_meta.stats["state"]["std"] = torch.tensor(
        norm_stats["norm_stats"]["state"]["std"][:num_motors], dtype=torch.float32
    )

    # Create LeRobot batch from Jax
    batch = {}
    batch[f"image"] = torch.from_numpy(example['observation/image']).permute(2,0,1) / 255.0
    batch[f"wrist_image"] = torch.from_numpy(example['observation/wrist_image']).permute(2,0,1) / 255.0
    batch["observation.state"] = torch.from_numpy(example["observation/state"])
    batch["action"] = torch.from_numpy(outputs["actions"])
    batch["task"] = example["prompt"]

    # Batchify
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].unsqueeze(0)
        elif isinstance(batch[key], str):
            batch[key] = [batch[key]]
        else:
            raise ValueError(f"{key}, {batch[key]}")

    # To device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device, dtype=torch.float32)

    noise = torch.from_numpy(noise).to(device=device, dtype=torch.float32)

    from lerobot.common import policies  # noqa

    cfg = PreTrainedConfig.from_pretrained(ckpt_torch_dir)
    cfg.pretrained_path = ckpt_torch_dir
    policy = make_policy(cfg, dataset_meta)

    # loss_dict = policy.forward(batch, noise=noise, time=time_beta)
    # loss_dict["loss"].backward()
    # print("losses")
    # display(loss_dict["losses_after_forward"])
    # print("pi_losses")
    # display(pi_losses)

    print(batch['image'].shape)
    print(batch['wrist_image'].shape)

    # actions = []
    # for _ in range(50):
    #     action = policy.select_action(batch, noise=noise)
    #     actions.append(action)
    # actions = torch.stack(actions, dim=1)
    actions, images, img_masks, state, lang_tokens, lang_masks = policy.select_action_chunk(batch, noise=noise)



    pi_actions = batch["action"]
    print("actions")
    display(actions)
    print()
    print("pi_actions")
    display(pi_actions)
    print("atol=3e-2", torch.allclose(actions, pi_actions, atol=3e-2))
    print("atol=2e-2", torch.allclose(actions, pi_actions, atol=2e-2))
    print("atol=1e-2", torch.allclose(actions, pi_actions, atol=1e-2))


    


if __name__ == "__main__":
    main()


'''
actions
Shape: torch.Size([1, 50, 8])
Mean: 0.10199584066867828
Std: 0.5305159687995911
Min: -0.7000712156295776
Max: 1.437532663345337

pi_actions
Shape: torch.Size([1, 50, 8])
Mean: 0.2686498463153839
Std: 1.0916320085525513
Min: -1.536781668663025
Max: 1.7205588817596436
atol=3e-2 False
atol=2e-2 False
atol=1e-2 False
'''