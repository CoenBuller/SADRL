#!/bin/bash
#SBATCH --job-name=smoke_test_marl
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/smoke_test_%j.out
#SBATCH --error=slurm_logs/smoke_test_%j.err

# Exit immediately if a command exits with a non-zero status
set -e

# Create log and output directories
mkdir -p slurm_logs
mkdir -p results_smoke_test

# Optional: Load CUDA modules if your cluster environment requires it
# module load cuda/12.2

# Prevent JAX from locking 90% of GPU memory upfront
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Starting MARL Smoke Test on node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "=================================================="

# Run through uv if present, otherwise default to python
if command -v uv &> /dev/null; then
    uv run python main.py --config smoke_test.yaml
else
    python main.py --config smoke_test.yaml
fi

echo "=================================================="
echo "Smoke test completed successfully!"
echo "=================================================="
