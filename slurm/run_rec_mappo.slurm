#!/bin/bash
#SBATCH --job-name=rec_mappo
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11


module purge
source /zfsstore/user/s2631415/SADRL/.venv/bin/activate
unset LD_LIBRARY_PATH
export PYTHONPATH=/zfsstore/user/s2631415/SocialJax:$PYTHONPATH
cd /zfsstore/user/s2631415/SADRL

nvidia-smi --query-gpu=name,memory.total --format=csv
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$MEM" -lt 30000 ]; then echo "GPU too small: ${MEM} MiB"; exit 1; fi

environments=(pd_arena, clean_up)
env=${environments[$((SLURM_ARRAY_TASK_ID % 4))]}
SEED=$((30 + SLURM_ARRAY_TASK_ID / 4))


python main.py --env $env --seed $SEED
