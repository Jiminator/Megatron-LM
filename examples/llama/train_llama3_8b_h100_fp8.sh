#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_fp8"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_fp8"}
TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

# Create directories if they don't exist
# mkdir -p "$(dirname "$CHECKPOINT_PATH")"
# mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=1
NUM_NODES=2
MASTER_ADDR=128.105.144.57
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters
TP_SIZE=1     
CP_SIZE=1     
PP_SIZE=1     
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2
NUM_LAYERS=8  
DTYPE="fp16"
SEQ_LENGTH=512
MAX_POSITION_EMBEDDINGS=512
# SEQ_LENGTH=8192
# MAX_POSITION_EMBEDDINGS=8192

# Data cache path (useful for both mock and real data)
# DATA_CACHE_PATH="${PWD}/benchmark_cache_llama3_8b_fp8"
# mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --rdzv_backend=c10d
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size 512
    --ffn-hidden-size 1792
    --num-attention-heads 4
    --group-query-attention
    --num-query-groups 1
    --kv-channels 16
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000 
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p 
    --untie-embeddings-and-output-weights
    --disable-bias-linear 
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 3
    --lr-decay-iters 2
    --lr-warmup-fraction .002
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4      # Specific to decoupled AdamW, ensure optimizer is compatible
    --decoupled-min-lr 4.5e-5  # Specific to decoupled AdamW
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --transformer-impl local
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
    --exit-duration-in-mins 235 
)

# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    # --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
# TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256" 
        # "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 128256"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 0
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    # --profile
    # --profile-step-start 1
    # --profile-step-end 3
    # --ckpt-format torch_dist 
    # --distributed-timeout-minutes 60
    # --save "$CHECKPOINT_PATH"
    # --load "$CHECKPOINT_PATH" 
    # --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

LOG_DIR="logs/"
mkdir -p $LOG_DIR
LOCAL_LOG="$LOG_DIR/node$NODE_RANK.log"
CONSOLIDATED_LOG="$LOG_DIR/all_nodes.log"

torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} | tee $LOCAL_LOG

# Post-processing: If this is node-2, send log to node-1 after training
if [ "$NODE_RANK" -eq 0 ]; then
    echo "Node 0 saving consolidated log at $CONSOLIDATED_LOG"
    cp "$LOCAL_LOG" "$CONSOLIDATED_LOG"
else
    echo "Node $NODE_RANK appending its log to node 0"
    ssh jimmys2@node-1 "cat >> $CONSOLIDATED_LOG" < "$LOCAL_LOG"
fi


# if [ "$NODE_RANK" -eq 0 ]; then
#     torchrun ${DISTRIBUTED_ARGS[@]} \
#         "$PRETRAIN_SCRIPT_PATH" \
#         ${MODEL_ARGS[@]} \
#         ${TRAINING_ARGS[@]} \
#         ${DTYPE_ARGS[@]} \
#         ${MODEL_PARALLEL_ARGS[@]} \
#         ${DATA_ARGS_LIST[@]} \
#         ${EVAL_AND_LOGGING_ARGS[@]} | tee "test/all_nodes.log"
# else
#     torchrun ${DISTRIBUTED_ARGS[@]} \
#         "$PRETRAIN_SCRIPT_PATH" \
#         ${MODEL_ARGS[@]} \
#         ${TRAINING_ARGS[@]} \
#         ${DTYPE_ARGS[@]} \
#         ${MODEL_PARALLEL_ARGS[@]} \
#         ${DATA_ARGS_LIST[@]} \
#         ${EVAL_AND_LOGGING_ARGS[@]} | ssh jimmys2@node-1 "cat >> /users/jimmys2/Megatron-LM/test/all_nodes.log"
# fi
set +x