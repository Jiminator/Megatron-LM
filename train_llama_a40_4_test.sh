#!/bin/bash

#SBATCH --job-name=megatron-multinode   # A descriptive name for your job
#SBATCH --account=bcrn-delta-gpu        # Your account
#SBATCH --partition=gpuA40x4            # The batch partition (e.g., gpuA40x4, not gpuA40x4-interactive)
#SBATCH --nodes=1                       # Number of nodes to use for training
#SBATCH --gpus-per-node=4               # Number of GPUs per node
#SBATCH --ntasks-per-node=4             # Number of processes per node (should match gpus-per-node)
#SBATCH --cpus-per-task=4              # Number of CPU cores per process
#SBATCH --mem=20g                       # Memory per node
#SBATCH --time=01:00:00                 # Walltime limit (HH:MM:SS)
#SBATCH --output=logs/a40_4_test/train_llama_a40_4_test_%j.out  # Standard output and error log


# --- Load Modules and Activate Environment ---
echo "Loading modules and activating environment..."
cd /projects/bcrn/jshong/Megatron-LM
module load cuda
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
module load nccl # loads the nccl built with the AWS nccl plugin for Slingshot11
module load anaconda3_gpu
module load gcc

source deactivate
source activate megatron

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

export LOG_HOME="logs/a40_4_test"
mkdir -p $LOG_HOME

# Memory Profiling
LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=1 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=1 NUM_LAYERS=4 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=1 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh 

LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=2 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=2 NUM_LAYERS=4 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=2 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh 

LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=4 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=4 NUM_LAYERS=4 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=1 MICRO_BATCH_SIZE=4 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh 

LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=1 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=1 NUM_LAYERS=2 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=1 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh 

LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=2 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=2 NUM_LAYERS=2 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=2 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh 

LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=4 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=4 NUM_LAYERS=2 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=2 MICRO_BATCH_SIZE=4 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh 

LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=1 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=1 NUM_LAYERS=1 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=1 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=2 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=2 NUM_LAYERS=1 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=2 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=4 NUM_LAYERS=0 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=4 NUM_LAYERS=1 STEP3=0 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/memory" TP_SIZE=4 MICRO_BATCH_SIZE=4 NUM_LAYERS=0 STEP3=1 bash examples/llama/train_llama3_8b_h100_fp8.sh


# Latency Profiling

LOG_DIR="$LOG_HOME/latency" TP_SIZE=1 MICRO_BATCH_SIZE=1 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/latency" TP_SIZE=1 MICRO_BATCH_SIZE=2 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/latency" TP_SIZE=1 MICRO_BATCH_SIZE=4 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 

LOG_DIR="$LOG_HOME/latency" TP_SIZE=2 MICRO_BATCH_SIZE=1 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/latency" TP_SIZE=2 MICRO_BATCH_SIZE=2 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/latency" TP_SIZE=2 MICRO_BATCH_SIZE=4 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 

LOG_DIR="$LOG_HOME/latency" TP_SIZE=4 MICRO_BATCH_SIZE=1 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/latency" TP_SIZE=4 MICRO_BATCH_SIZE=2 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 
LOG_DIR="$LOG_HOME/latency" TP_SIZE=4 MICRO_BATCH_SIZE=4 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh 


# Intermediate timing

LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 MICRO_BATCH_SIZE=2 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 MICRO_BATCH_SIZE=2 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 MICRO_BATCH_SIZE=2 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 MICRO_BATCH_SIZE=4 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 MICRO_BATCH_SIZE=4 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=1 MICRO_BATCH_SIZE=4 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh


LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 MICRO_BATCH_SIZE=2 NUM_LAYERS=32 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 MICRO_BATCH_SIZE=2 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 MICRO_BATCH_SIZE=2 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 MICRO_BATCH_SIZE=2 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 MICRO_BATCH_SIZE=4 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 MICRO_BATCH_SIZE=4 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=2 MICRO_BATCH_SIZE=4 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 NUM_LAYERS=32 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 MICRO_BATCH_SIZE=2 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 MICRO_BATCH_SIZE=2 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 MICRO_BATCH_SIZE=2 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh

LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 MICRO_BATCH_SIZE=4 NUM_LAYERS=16 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 MICRO_BATCH_SIZE=4 NUM_LAYERS=8 bash examples/llama/train_llama3_8b_h100_fp8.sh
LOG_DIR="$LOG_HOME/interm" TP_SIZE=4 MICRO_BATCH_SIZE=4 NUM_LAYERS=4 bash examples/llama/train_llama3_8b_h100_fp8.sh
