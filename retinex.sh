#!/bin/bash
#SBATCH --job-name=darkage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=darkage%j.log
#SBATCH --error=darkage%j.log


CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate loco

echo "running host: $(hostname)"
echo "checking gpu status: $(nvidia-smi)"
echo "start training..."

python -W ignore predict_retinex_updated_colab.py --batch_size 32
