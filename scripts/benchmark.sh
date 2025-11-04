#!/bin/bash
# Run scaling experiments to measure performance across different GPU counts

echo "=========================================="
echo "GPU Scaling Benchmark"
echo "=========================================="
echo "This will test training with 1, 2, and 4 GPUs"
echo ""

# Create results directory
mkdir -p results

# Function to run benchmark
run_benchmark() {
    local gpus=$1
    local output_file="results/benchmark_${gpus}gpu.log"
    
    echo "Testing with $gpus GPU(s)..."
    
    torchrun \
        --nproc_per_node=$gpus \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        src/ddp_trainer.py \
        --batch-size 128 \
        --epochs 5 \
        --monitor \
        2>&1 | tee $output_file
    
    echo "✓ Completed $gpus GPU test"
    echo ""
}

# Run benchmarks
run_benchmark 1
sleep 2
run_benchmark 2
sleep 2
run_benchmark 4

echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to results/ directory"
echo ""
echo "To analyze results:"
echo "  python analyze_results.py"