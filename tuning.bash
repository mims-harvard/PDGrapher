#!/bin/bash
#SBATCH -c 1
#SBATCH -t 5-00:00
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o examples/tuning_%j.out
#SBATCH -e examples/tuning_%j.err

module load python/3.10.11
module load gcc/9.2.0 cuda/11.7

source "venv/bin/activate"
/n/cluster/bin/job_gpu_monitor.sh & python3 examples/hyperparameter_tuning.py
