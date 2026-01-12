# add conda path /mnt/lustre-grete/usr/u12045/projects/LLAVA-Med/envs/lerobot
export PATH=/mnt/lustre-grete/usr/u12045/projects/LLAVA-Med/envs/lerobot/bin:$PATH
export HF_HOME=/mnt/lustre-grete/usr/u12045/vla/hf_cache
export TMPDIR=/mnt/lustre-grete/usr/u12045/vla/cache
export PYTHONPATH=/mnt/lustre-grete/usr/u12045/vla/duci/pi0_lerobo:$PYTHONPATH



cp configs/policy_config/default_haoming_fractal.json /mnt/lustre-grete/usr/u12045/vla/hf_cache/hub/models--lerobot--pi0/snapshots/8f50aacbe079a026391616cf22453de528f2a873/config.json
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29500 lerobot/scripts/train_accelerate.py \
  --policy.path=lerobot/pi0 \
  --dataset.repo_id=IPEC-COMMUNITY/fractal20220817_data_lerobot \
  --output_dir=outputs/train/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_test \
  --job_name=test \
  --config_path=configs/fractal_config/default.json \
  --batch_size=1 \
  --policy.gradient_accumulation_steps=1 \
  --log_freq=10 \
  --wandb.enable=false


# resume_dir="outputs/train/2025-06-28/20-28-43_fractal_finetune_libero_config_chunk4"
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 29500 lerobot/scripts/train_accelerate.py \
#   --dataset.repo_id=IPEC-COMMUNITY/fractal20220817_data_lerobot \
#   --resume=true \
#   --output_dir="$resume_dir" \
#   --config_path="$resume_dir/checkpoints/last/pretrained_model/train_config.json" \
