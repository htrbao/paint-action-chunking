export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_NO_ADVISORY_LOCK=1
export HF_HUB_DISABLE_TELEMETRY=1
export PATH=/home/binhng/conda_setup/miniconda3/envs/pi0/bin:$PATH
export HF_HOME=/mnt/data/sftp/data/vla_intern/workspace/hf_home
export TMPDIR=/mnt/data/sftp/data/vla_intern/workspace/cache
export PYTHONPATH=/mnt/data/sftp/data/vla_intern/workspace/binh/VLA-Humanoid:$PYTHONPATH

CHECKPOINT_DIR=/mnt/data/sftp/data/vla_intern/workspace/binh/VLA-Humanoid/cpkt/temp/baseline
DATA_DIR=/mnt/data/sftp/data/vla_intern/workspace/data/ROBOCASA/robocasa_30_demos_lerobot_5_chosen_tasks_v3
EXP_NAME=pi0_base_robocasa_30demos
SAVE_CHECKPOINT_DIR=/mnt/data/sftp/data/vla_intern/workspace/binh/VLA-Humanoid

POLICY_CONFIG_PATH=configs/robocasa_config/policy_config.json
DATA_CONFIG_PATH=configs/robocasa_config/data_config.json



cp $POLICY_CONFIG_PATH $CHECKPOINT_DIR/config.json
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29800 lerobot/scripts/train_accelerate.py \
  --policy.path=$CHECKPOINT_DIR \
  --dataset.root=$DATA_DIR \
  --output_dir=outputs/train/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_$EXP_NAME \
  --job_name=$EXP_NAME \
  --config_path=$DATA_CONFIG_PATH \
  --batch_size=12 \
  --steps=100000 \
  --policy.gradient_accumulation_steps=1 \
  --wandb.mode=online \
  --log_freq=10 