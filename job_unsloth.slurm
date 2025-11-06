#!/bin/bash
#SBATCH --job-name=                              # Job Name
#SBATCH --account=                               # Account
#SBATCH --partition=                             # Partition
#SBATCH --nodes=                                 # Number of nodes
#SBATCH --ntasks-per-node=                       # Number of tasks per node
#SBATCH --gres=gpu:                              # Number of GPUs
#SBATCH --cpus-per-task=                         # Number of CPU cores per task
#SBATCH --mem=                                   # Memory per node
#SBATCH --time=                                  # Max run time (hh:mm:ss)
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err

# ====== Environment Setup ======
WORKDIR="$HOME/training-gptoss"
ENVNAME="env_training"
LOGDIR="$WORKDIR/logs"
RESULTDIR="$WORKDIR/results"

mkdir -p "$LOGDIR" "$RESULTDIR"

# ====== Conda Initialization ======
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVNAME"
echo "Activated conda environment: $ENVNAME"

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

srun python "$WORKDIR/train_unsloth.py"

echo "=== Job finished ==="