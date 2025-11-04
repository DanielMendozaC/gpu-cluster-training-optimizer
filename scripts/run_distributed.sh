#!/bin/bash
# Launch distributed training with torchrun

# Default settings
GPUS=${1:-4}  # Number of GPUs (default: 4)
BATCH_SIZE=${2:-128}  # Batch size per GPU
EPOCHS=${3:-10}

echo "=========================================="
echo "Launching Distributed Training"
echo "=========================================="
echo "GPUs: $GPUS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((GPUS * BATCH_SIZE))"
echo "Epochs: $EPOCHS"
echo "=========================================="
echo ""

# Set NCCL environment variables for optimization
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0

# Launch with torchrun
torchrun \
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    src/ddp_trainer.py \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --monitor

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="