#!/bin/bash
#SBATCH --job-name=                      # Job Name
#SBATCH --account=                       # Account
#SBATCH --partition=                     # Partition
#SBATCH --nodes=                         # Number of nodes
#SBATCH --ntasks-per-node=               # Number of node for per task
#SBATCH --gres=gpu:                      # Number of GPU
#SBATCH --cpus-per-task=                 # Number of CPU
#SBATCH --mem=                           # The size of memory 
#SBATCH --time=                          # Max run time 
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err


# ====== Environment Setup ======

WORKDIR=$(pwd)
ENVNAME="env_training"
ENVDIR="$WORKDIR/$ENVNAME"
LOGDIR="$HOME/training-gptoss/logs"
RESULTDIR="$HOME/training-gptoss/results"

mkdir -p "$LOGDIR" "$RESULTDIR"

source "$ENVDIR/bin/activate"
echo "Activated virtual environment."

which python
python -c "import torch; print('Torch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# ====== Happy Training ======

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=$((12000 + RANDOM % 10000))
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

echo "Starting training..."

srun "$ENVDIR/bin/python" "$WORKDIR/train.py"

echo "=== Job finished. ==="