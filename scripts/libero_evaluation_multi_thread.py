import os
os.environ["PATH"] = "/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/miniconda3/envs/pitorch/bin:" + os.environ.get("PATH", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/hf_cache"
os.environ["PYTHONPATH"] = "/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/binh/VLA-Humanoid:" + os.environ.get("PYTHONPATH", "")
import sys
sys.path.insert(0, "/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/duc3/VLA-Humanoid")

sys.path.insert(0, "/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/binh/VLA-Humanoid")

import collections
import dataclasses
import logging
import math
import pathlib
import multiprocessing as mp  # added for parallel workers

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tqdm
import tyro

from myutils.pi0_infer import Pi0TorchInference, normalize_gripper_action, invert_gripper_action

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

'''
python scripts/libero_evaluation_multi_thread.py \
--exp_name=libero_spatial_100%_12_layers \
--task_suite_name=libero_10
'''
@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters%
    #################################################################################################################
    pretrained_model_path: str = "/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/binh/VLA-Humanoid/outputs/train/2025-11-26/11-34-23_libero_100%_base_12layers_only/checkpoints/060000/pretrained_model"
    resize_size: int = 224
    replan_steps: int = 10

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    exp_name: str = "test"
    # -------------------------------------------------------------------------------
    # Multi-GPU parallelization parameters
    # -------------------------------------------------------------------------------
    gpus: str = "0,1,2,3"  # comma-separated list of GPU IDs to use


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Determine if we're in a worker with a rank/world_size set
    rank = getattr(args, "rank", None)
    world_size = getattr(args, "world_size", None)
    logging.info(f"world_size: {world_size}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Rank: {rank} | Task suite: {args.task_suite_name} | replan_steps: {args.replan_steps} | {args.pretrained_model_path}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    import time
    if rank == 1:
        time.sleep(30)
    if rank == 2:
        time.sleep(60)
    if rank == 3:
        time.sleep(90)
    try:
        mypolicy = Pi0TorchInference(args.pretrained_model_path, device=f"cuda:0")
        logging.info(f"Rank {rank} | Successfully loaded policy")
        print(f"[GPU{rank}] Loaded policy successfully")
    except Exception as e:
        logging.info(f"Rank {rank} | Failed to load policy: {e}")
        print(f"[GPU{rank}] Failed to load policy: {e}")
        return
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        logging.info(f"Task_id: {task_id}")

        # if running in parallel, only process your share
        if world_size is not None and (task_id % world_size) != rank:
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            if task_episodes % 10 == 0:
                logging.info(f"Task_id: {task_id} | Starting episode {task_episodes+1}... | {task_description}")
            while t < max_steps + args.num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # img = image_tools.convert_to_uint8(
                #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                # )
                # wrist_img = image_tools.convert_to_uint8(
                #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                # )

                # Save preprocessed image for replay video
                replay_images.append(img)

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
                    element = {
                        "observation.images.image": img,
                        "observation.images.wrist_image": wrist_img,
                        "observation.state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "task": str(task_description),
                    }

                    # Query model to get action
                    action_chunk = mypolicy.libero_step(element)
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)  # for libero, -1=open, 1=close

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

        # Log current results
        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def worker(gpu_id: int, rank: int, world_size: int, args: Args):
    """Worker process: bind to one GPU and run eval_libero on your shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # import torch
    # assert torch.cuda.is_available(), f"[GPU{rank}] CUDA not available"
    # print(f"[GPU{rank}] CUDA device count: {torch.cuda.device_count()}")
    # print(f"[GPU{rank}] Current device: {torch.cuda.current_device()}")

    args.rank = rank
    args.world_size = world_size

    # 1) ensure log directory exists
    log_dir = pathlib.Path(f"libero_eval_logs/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2) clear any existing handlers and install a FileHandler
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    log_file = log_dir / f"eval_gpu{gpu_id}_{args.task_suite_name}.log"
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [GPU %(name)s] %(levelname)s: %(message)s")
    )
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    eval_libero(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)

    # parse GPU list and spawn one worker per GPU
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip().isdigit()]
    world_size = len(gpu_list)
    procs = []
    for rank, gpu_id in enumerate(gpu_list):
        p = mp.Process(target=worker, args=(gpu_id, rank, world_size, args), name=f"GPU{gpu_id}")
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
