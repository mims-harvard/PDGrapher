#!/bin/bash
#SBATCH -c 2
#SBATCH -t 5-00:00
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o examples/%j.out
#SBATCH -e examples/%j.err

module load python/3.10.11
module load gcc/9.2.0 cuda/11.7

source "venv/bin/activate"
/n/cluster/bin/job_gpu_monitor.sh & python3 examples/multiple_folds.py
