#!/bin/bash
#SBATCH -c 1
#SBATCH -t 1-02:00
#SBATCH -p gpu0
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH -o tests/test_%j.out
#SBATCH -e tests/est_%j.err

module load python/3.10.11
module load gcc/9.2.0 cuda/11.7

source "venv/bin/activate"
/n/cluster/bin/job_gpu_monitor.sh & python3 tests/test_package.py
