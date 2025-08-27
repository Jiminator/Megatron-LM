#!/bin/bash

#SBATCH --job-name=megatron-multinode   # A descriptive name for your job
#SBATCH --account=bcrn-delta-gpu        # Your account
#SBATCH --partition=gpuA40x4            # The batch partition (e.g., gpuA40x4, not gpuA40x4-interactive)
#SBATCH --nodes=1                       # Number of nodes to use for training
#SBATCH --gpus-per-node=2               # Number of GPUs per node
#SBATCH --ntasks-per-node=2             # Number of processes per node (should match gpus-per-node)
#SBATCH --cpus-per-task=16              # Number of CPU cores per process
#SBATCH --mem=20g                       # Memory per node
#SBATCH --time=00:05:00                 # Walltime limit (HH:MM:SS)
#SBATCH --output=logs/megatron-%j.out   # Path to the output log file (make sure 'logs' directory exists)

# --- Load Modules and Activate Environment ---
echo "Loading modules and activating environment..."
cd /projects/bcrn/jshong/Megatron-LM || exit 1
module load cuda anaconda3_gpu gcc
source deactivate
source activate megatron

# # --- Set Up Distributed Training Environment ---
# # MASTER_ADDR is the IP address of the first node in the job
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# # MASTER_PORT can be a fixed number, but ensure it's not used by other jobs
# export MASTER_PORT=6000


# echo "NODELIST=${SLURM_NODELIST}"
# echo "MASTER_ADDR=${MASTER_ADDR}"
# echo "MASTER_PORT=${MASTER_PORT}"
# echo "Running on ${SLURM_NNODES} nodes with ${SLURM_NTASKS} total processes."

echo "NODELIST=${SLURM_NODELIST}" 
export NUM_NODES=${NUM_NODES:-$SLURM_NNODES}
echo "NUM_NODES=${NUM_NODES}"
export GPUS_PER_NODE=${GPUS_PER_NODE:-$SLURM_GPUS_ON_NODE}
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
echo "MASTER_ADDR=${MASTER_ADDR}"
export MASTER_PORT=${MASTER_PORT:-6000}
echo "MASTER_PORT=${MASTER_PORT}"
export NODE_RANK=${SLURM_PROCID}
echo "NODE_RANK=${NODE_RANK}"
export WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
echo "WORLD_SIZE=${WORLD_SIZE}"
export NUM_LAYERS=2 
echo "NUM_LAYERS=${NUM_LAYERS}"

# --- Run the Training Script ---
# Use `srun` to launch the training script on each allocated task.
# Slurm automatically provides `SLURM_PROCID` (rank) and `SLURM_NTASKS` (world_size)
# to each process. Your training script should be written to use these variables.
echo "Launching training script..."

srun --cpu-bind=none --label bash examples/llama/train_llama3_8b_h100_fp8_slurm.sh

echo "Job finished."