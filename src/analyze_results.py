#!/usr/bin/env python3
"""
Analyze benchmark results and generate performance statistics
"""

import re
import json
from pathlib import Path


def parse_benchmark_log(log_file):
    """Parse benchmark log file and extract metrics"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract GPU count from filename
    match = re.search(r'(\d+)gpu', log_file)
    gpu_count = int(match.group(1)) if match else 1
    
    # Extract total training time
    time_match = re.search(r'Total time: ([\d.]+)s', content)
    total_time = float(time_match.group(1)) if time_match else None
    
    # Extract throughput samples
    throughputs = re.findall(r'Throughput: ([\d.]+) img/s', content)
    throughputs = [float(t) for t in throughputs]
    
    # Extract GPU utilization from monitoring
    gpu_utils = re.findall(r'GPU Util: ([\d.]+)%', content)
    gpu_utils = [float(u) for u in gpu_utils]
    
    # Extract accuracy
    acc_match = re.search(r'Best accuracy: ([\d.]+)%', content)
    best_acc = float(acc_match.group(1)) if acc_match else None
    
    return {
        'gpu_count': gpu_count,
        'total_time': total_time,
        'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
        'max_throughput': max(throughputs) if throughputs else 0,
        'avg_gpu_util': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
        'best_accuracy': best_acc
    }


def calculate_scaling_metrics(results):
    """Calculate speedup and efficiency"""
    if not results:
        return []
    
    # Use 1 GPU as baseline
    baseline = next((r for r in results if r['gpu_count'] == 1), None)
    if not baseline or not baseline['total_time']:
        return results
    
    baseline_time = baseline['total_time']
    
    for result in results:
        if result['total_time']:
            result['speedup'] = baseline_time / result['total_time']
            result['efficiency'] = (result['speedup'] / result['gpu_count']) * 100
        else:
            result['speedup'] = 0
            result['efficiency'] = 0
    
    return results


def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Header
    print(f"{'GPUs':<8}{'Time (s)':<12}{'Throughput':<15}{'Speedup':<12}{'Efficiency':<12}{'Accuracy':<12}")
    print("-"*80)
    
    # Data rows
    for r in sorted(results, key=lambda x: x['gpu_count']):
        print(f"{r['gpu_count']:<8}"
              f"{r['total_time']:<12.1f}"
              f"{r['avg_throughput']:<15.0f}"
              f"{r['speedup']:<12.2f}x"
              f"{r['efficiency']:<12.1f}%"
              f"{r.get('best_accuracy', 0):<12.1f}%")
    
    print("="*80 + "\n")


def main():
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("Error: results/ directory not found. Run ./benchmark.sh first.")
        return
    
    # Parse all benchmark logs
    results = []
    for log_file in results_dir.glob('benchmark_*gpu.log'):
        print(f"Parsing {log_file.name}...")
        result = parse_benchmark_log(str(log_file))
        results.append(result)
    
    if not results:
        print("No benchmark results found.")
        return
    
    # Calculate scaling metrics
    results = calculate_scaling_metrics(results)
    
    # Print results
    print_results_table(results)
    
    # Save to JSON
    output_file = results_dir / 'benchmark_summary.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    
    # Print key insights
    if len(results) >= 2:
        print("\nKEY INSIGHTS:")
        max_gpu_result = max(results, key=lambda x: x['gpu_count'])
        print(f"  • Scaling efficiency at {max_gpu_result['gpu_count']} GPUs: {max_gpu_result['efficiency']:.1f}%")
        print(f"  • Peak throughput: {max_gpu_result['max_throughput']:.0f} images/second")
        print(f"  • Average GPU utilization: {max_gpu_result['avg_gpu_util']:.1f}%")


if __name__ == '__main__':
    main()