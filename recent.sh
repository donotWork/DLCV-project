#!/bin/bash
#SBATCH --job-name=pretrained_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate loco

echo "Running on host: $(hostname)"
nvidia-smi

echo "Starting training..."
python3 -W ignore recent.py \
    --data /home/gautamarora/project/recent.yaml \
    --model yolov8m.pt \
    --epochs 50 \
    --batch 8 \
    --imgsz 1280 720 \
    --device 0
