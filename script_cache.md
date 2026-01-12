srun --pty --job-name=pi0 \
     --partition=main \
     --nodes=1 \
     --ntasks=1 \
     --gpus=nvidia_h100_80gb_hbm3:4 \
     --cpus-per-task=32 \
     --mem=80G \
     --time=48:00:00 \
     bash -i


bash scripts/debug.sh

bash scripts/ft_robocasa_30.sh
bash scripts/ft_robocasa_100.sh