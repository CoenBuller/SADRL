#!/bin/bash
# Generic short GPU job: runs `python -u <args>` inside ~/socialjax (used by `./alice.sh gpu-test`).
#SBATCH --job-name=sj-python
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
module load "${PYTHON_MODULE:-Python/3.10.8-GCCcore-12.2.0}"
source "$HOME/${VENV_DIR:-socialjax/venv}/bin/activate"
cd "$HOME/socialjax"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo "Job $SLURM_JOB_ID on $(hostname) | partition $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u "$@"
