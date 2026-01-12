import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
from moviepy.editor import ImageSequenceClip
import copy

# This is for using the locally installed repo clone when using slurm
from calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel

# sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)       
from calvin.calvin_models.calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

import json
import numpy as np

import multiprocessing  # for process info
import asyncio
import json
import numpy as np

from myutils.pi0_infer import *
import multiprocessing as mp  # added for parallel workers


logger = logging.getLogger(__name__)

np.float = float

def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)
    # insert your own env wrapper if needed
    return env

def proc_print(*args, **kwargs):
    """
    Helper function to print messages with process name and PID.
    """
    proc = multiprocessing.current_process()
    print(f"[{proc.name} (pid: {proc.pid})]", *args, flush=True, **kwargs)

def setup_process_logging(eval_log_dir, start, end):
    """
    Redirects stdout and stderr to a unique file for this process.
    All prints will then be written to this file.
    """
    proc = multiprocessing.current_process()
    # Create a unique filename using process name and pid.
    log_filename = f"{eval_log_dir}/eval_log_{start}_{end}.log"
    log_filepath = os.path.join(os.getcwd(), log_filename)
    f = open(log_filepath, "a", buffering=1)  # line-buffered
    sys.stdout = f
    sys.stderr = f
    proc_print("Logging redirected to", log_filepath)

def evaluate_policy(args, model, env, epoch=-1, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.
    """
    conf_dir = Path("calvin/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    # eval_log_dir = get_log_dir(eval_log_dir)
    results = []
    with open("calvin/eval_sequences.json", "r") as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[:args.num_sequences]

    plans = defaultdict(list)

    # Prepend process info to tqdm description.
    proc = multiprocessing.current_process()
    tqdm_prefix = f"[{proc.name} (pid: {proc.pid})] "

    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True, desc=tqdm_prefix)

    current_seq_idx_for_each_rank = 0
    for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if args.world_size is not None and (seq_idx % args.world_size) != args.rank:
            continue
        result = evaluate_sequence(args, env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans)
        results.append(result)
        if not args.debug:
            res_str = f"Global sequence idx: {seq_idx+1} | Local sequence idx {current_seq_idx_for_each_rank+1} | Local Sum results: {sum(results)} |" + " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            logging.info(res_str)
            # eval_sequences.set_description(res_str)
        current_seq_idx_for_each_rank += 1

    if create_plan_tsne:
        create_tsne(plans, args.eval_log_dir, epoch)
    print_and_save(results, eval_sequences, Path(args.eval_log_dir), epoch)

    return results

def evaluate_sequence(args, env, model, task_checker, initial_state, eval_sequence, val_annotations, plans):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if args.debug:
        time.sleep(1)
        print()
        print()
        proc_print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="", flush=True)
    for subtask in eval_sequence:
        success = rollout(args, env, model, task_checker, subtask, val_annotations, plans)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(args, env, model, task_oracle, subtask, val_annotations, plans):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if args.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
        img_list = []
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    logging.info(f"Subtask {subtask} - {lang_annotation}")
    start_info = env.get_info()

    for step in range(0, args.ep_len//args.action_horizon):
        ### Reformat obs
        obs = {
            'image': obs["rgb_obs"]["rgb_static"],
            'wrist_image':obs["rgb_obs"]["rgb_gripper"],
            "state": obs["robot_obs"],
            "task": str(lang_annotation),
        }
        action_chunk = model.calvin_step(obs)
        action_chunk[:, -1] = np.where(action_chunk[:, -1] > 0, 1, -1)
        for action in action_chunk[:args.action_horizon]:
            action = {"action": action, "type": "joint_abs"}
            obs, _, _, current_info = env.step(action)

            if args.debug:
                img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
                img_list.append(img_copy)
                # time.sleep(0.1)
            if step == 0:
                # for tsne plot, only if available
                collect_plan(model, plans, subtask)
    
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if args.debug:
                    print(colored("success", "green"), end=" ")
                    clip = ImageSequenceClip(img_list, fps=30)
                    clip.write_gif(
                        os.path.join(
                            args.eval_log_dir, f"{args.prefix_gif}_{subtask}-succ.gif"
                        ),
                        fps=30,
                    )
                return True
        
    if args.debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(
            os.path.join(args.eval_log_dir, f"{args.prefix_gif}_{subtask}-fail.gif"),
            fps=30,
        )
    return False


import pathlib
def worker(gpu_id: int, rank: int, world_size: int, args):
    """Worker process: bind to one GPU and run eval_libero on your shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    args.rank = rank
    args.world_size = world_size

    # 1) ensure log directory exists
    log_dir = pathlib.Path(f"{args.eval_log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2) clear any existing handlers and install a FileHandler
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    log_file = log_dir / f"eval_gpu{rank}.log"
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [GPU %(name)s] %(levelname)s: %(message)s")
    )
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


    import time
    time.sleep(rank * 10)
    try:
        if args.infer_version == 'jax':
            ckpt_jax_dir = Path(f"/mnt/lustre-grete/usr/u12045/vla/duci/openpi/checkpoints/pi0_calvin_50%_joint/pi0_calvin_50%_joint/30000")
            dataset_repo_id = "ducido/calvin_task_D_D_scale_50_lerobo_format"
            with open(ckpt_jax_dir / f"assets/{dataset_repo_id}/norm_stats.json") as f:
                norm_stats = json.load(f)

            model = Pi0JaxInference(
                model_dir=args.model_path,
                dataset_repo_id=dataset_repo_id,
                norm_stats=norm_stats,
                device='cuda:0'
            )
        elif args.infer_version == 'torch':
            model = Pi0TorchInference(
                model_dir=args.model_path,
                device='cuda:0'
            )
        logging.info(f"Rank {rank} | Successfully loaded policy version {args.infer_version} | {args.model_path}")
        print(f"[GPU{rank}] Loaded policy successfully")
    except Exception as e:
        logging.info(f"Rank {rank} | Failed to load policy: {e}")
        print(f"[GPU{rank}] Failed to load policy: {e}")
        return


    env = make_env(args.dataset_path)
    evaluate_policy(args, model, env)



def main():
    seed_everything(5, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.", default="/my_data/CALVIN/all_scale_D/scale_100")

    # arguments for loading default model
    parser.add_argument("--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir.")
    parser.add_argument("--checkpoints", type=str, default=None, help="Comma separated list of epochs for which checkpoints will be loaded")
    parser.add_argument("--avail_gpus", type=str, default=None, help="number of gpus to use for evaluation")
    parser.add_argument("--last_k_checkpoints", type=int, help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.")

    # arguments for loading custom model or custom language embeddings
    parser.add_argument("--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture.")
    parser.add_argument("--action_horizon", default=10, type=int, help="")
    parser.add_argument("--ep_len", default=360, type=int, help="")
    parser.add_argument("--num_sequences", default=1000, type=int, help="")
    parser.add_argument("--model_path", required=True, help="")
    parser.add_argument("--infer_version", default="torch", help="Use torch or jax inference code")
    parser.add_argument("--num_processes_per_gpu", type=int, default=1, help="Number of parallel processes to run per GPU")

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.", default=False)
    parser.add_argument("--eval_log_dir", type=str, help="Where to log the evaluation results.", default="./calvin_eval_logs/default_eval")
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    gpu_list = [int(x) for x in args.avail_gpus.split(",") if x.strip().isdigit()]
    world_size = len(gpu_list) * args.num_processes_per_gpu
    procs = []

    for rank in range(world_size):
        gpu_id = gpu_list[rank % len(gpu_list)]  # cycle through available GPUs
        p = mp.Process(target=worker, args=(gpu_id, rank, world_size, args), name=f"GPU{gpu_id}_proc{rank}")
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
if __name__ == "__main__":
    main()
    sys.exit(0)