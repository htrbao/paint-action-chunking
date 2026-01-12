#!/bin/bash
#SBATCH -p grete,grete-h100,kisski,kisski-h100,grete:shared,grete-h100:shared
#SBATCH --job-name=pi0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=80GB
#SBATCH -t 48:00:00


export PATH=/mnt/lustre-grete/usr/u12045/projects/LLAVA-Med/envs/lerobot/bin:$PATH
export HF_HOME=/mnt/lustre-grete/usr/u12045/vla/hf_cache
export TMPDIR=/mnt/lustre-grete/usr/u12045/vla/cache
export PYTHONPATH=/mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid:$PYTHONPATH


PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


exp_name="100%_defaultconfig_removepaddingloss_uplr_true"
python scripts/calvin_evaluation_mt.py \
    --dataset_path /mnt/lustre-grete/usr/u12045/vla/hf_cache/h5_calvin/all_scale_D/scale_100 \
    --eval_log_dir calvin_eval_logs/$exp_name \
    --model_path  $PROJECT_DIR/outputs/train/2025-07-30/18-05-43_calvin_100%_defaultconfig_removepaddingloss_bs56_lr2e-4_decay2e-5/checkpoints/080000/pretrained_model \
    --action_horizon 10 \
    --num_sequences 1000 \
    --avail_gpus '0' \
    --num_processes_per_gpu 10


# # checkpoint converted from Jax
# exp_name="50%_jaxcp_a50_again"
# python scripts/calvin_evaluation_mt.py \
#     --dataset_path /mnt/lustre-grete/usr/u12045/vla/hf_cache/h5_calvin/all_scale_D/scale_100 \
#     --eval_log_dir calvin_eval_logs/$exp_name \
#     --model_path  /mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid/calvin_jaxcp_conversion_to_torch \
#     --action_horizon 10 \
#     --num_sequences 1000 \
#     --avail_gpus '0' \
#     --infer_version 'jax' \
#     --num_processes_per_gpu 5
