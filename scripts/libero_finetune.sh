# export HF_HOME=/data/hf_cache
# export TMPDIR=/data/cache
# export PYTHONPATH=/workspace/VLAGuidance:$PYTHONPATH
# CHECKPOINT_DIR=/data/temp/baseline
# DATA_DIR=/data/merged_libero_scale_100_mask_depth_noops_lerobot
# EXP_NAME=libero_100%_test
# SAVE_CHECKPOINT_DIR=/data
# POLICY_CONFIG_PATH=configs/policy_config/default_decay300k.json
# OTHER_CONFIG_PATH=configs/libero_config/default.json


# ############ Path for nhr
# export PATH=/mnt/lustre-grete/usr/u12045/projects/LLAVA-Med/envs/lerobot/bin:$PATH
# export HF_HOME=/mnt/lustre-grete/usr/u12045/vla/hf_cache
# export TMPDIR=/mnt/lustre-grete/usr/u12045/vla/cache
# export PYTHONPATH=/mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid:$PYTHONPATH
# CHECKPOINT_DIR=/mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid/pi0_torch_newcp
# DATA_DIR=/mnt/lustre-grete/usr/u12045/vla/hf_cache/lerobot/binhng/merged_libero_mask_depth_noops_lerobot_40
# EXP_NAME=libero_40%_defaultconfig
# SAVE_CHECKPOINT_DIR=/mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid

# # ############ Path for kisski
export PATH=/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/miniconda3/envs/pitorch/bin:$PATH
export HF_HOME=$PROJECT_DIR/VLA/hf_cache
export TMPDIR=$PROJECT_DIR/VLA/cache
export PYTHONPATH=$PROJECT_DIR/VLA/binh/VLA-Humanoid:$PYTHONPATH
CHECKPOINT_DIR=$PROJECT_DIR/VLA/temp/baseline
DATA_DIR=$PROJECT_DIR/VLA/LIBERO/merged_libero_scale_100_mask_depth_noops_lerobot
EXP_NAME=libero_100%_base_12layers_only
SAVE_CHECKPOINT_DIR=$PROJECT_DIR/VLA/binh/VLA-Humanoid
POLICY_CONFIG_PATH=configs/policy_config/default_decay300k.json
OTHER_CONFIG_PATH=configs/libero_config/default.json


# ## For debug
# cp configs/policy_config/default.json $CHECKPOINT_DIR/config.json
# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port 29800 lerobot/scripts/train_accelerate.py \
#   --policy.path=$CHECKPOINT_DIR \
#   --dataset.root=$DATA_DIR \
#   --output_dir=outputs/train/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_$EXP_NAME \
#   --job_name=$EXP_NAME \
#   --config_path=configs/libero_config/default.json \
#   --batch_size=5 \
#   --policy.gradient_accumulation_steps=1 \
#   --log_freq=10 \
#   --wandb.enable=false

cp $POLICY_CONFIG_PATH $CHECKPOINT_DIR/config.json
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29800 lerobot/scripts/train_accelerate.py \
  --policy.path=$CHECKPOINT_DIR \
  --dataset.root=$DATA_DIR \
  --output_dir=$SAVE_CHECKPOINT_DIR/outputs/train/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_$EXP_NAME \
  --job_name=$EXP_NAME \
  --config_path=$OTHER_CONFIG_PATH \
  --batch_size=18 \
  --policy.gradient_accumulation_steps=1 \
  --save_freq=10000 \
  --log_freq=10 \
  --wandb.mode=offline

# resume_dir="/data/outputs/train/2025-09-18/08-24-29_libero_100%_test"
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=4 --main_process_port 29500 lerobot/scripts/train_accelerate.py \
#   --resume=true \
#   --output_dir="$resume_dir" \
#   --config_path="$resume_dir/checkpoints/last/pretrained_model/train_config.json"