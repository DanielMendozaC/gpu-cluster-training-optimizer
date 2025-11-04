#!/usr/bin/env python3
"""
GPU Monitoring utilities for distributed training
Tracks GPU utilization, memory, throughput, and communication overhead
"""

import os
import time
import subprocess
import threading
from collections import deque
import json


class GPUMonitor:
    """Real-time GPU monitoring for distributed training"""
    
    def __init__(self, rank, log_file='gpu_metrics.json'):
        self.rank = rank
        self.log_file = f'rank_{rank}_{log_file}'
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
        
    def get_gpu_stats(self):
        """Get GPU statistics using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            gpu_stats = []
            
            for line in lines:
                parts = [x.strip() for x in line.split(',')]
                if len(parts) == 5:
                    gpu_stats.append({
                        'gpu_id': int(parts[0]),
                        'utilization': float(parts[1]),
                        'memory_used': float(parts[2]),
                        'memory_total': float(parts[3]),
                        'temperature': float(parts[4])
                    })
            
            return gpu_stats
        except Exception as e:
            print(f"Warning: Could not get GPU stats: {e}")
            return []
    
    def log_metrics(self, epoch, batch, loss, throughput):
        """Log training metrics"""
        gpu_stats = self.get_gpu_stats()
        
        # Get stats for current GPU (rank)
        current_gpu = None
        if self.rank < len(gpu_stats):
            current_gpu = gpu_stats[self.rank]
        
        metric = {
            'timestamp': time.time(),
            'epoch': epoch,
            'batch': batch,
            'rank': self.rank,
            'loss': loss,
            'throughput': throughput,
            'gpu': current_gpu
        }
        
        self.metrics_history.append(metric)
        
        # Save periodically
        if len(self.metrics_history) % 100 == 0:
            self.save_metrics()
    
    def start_background_monitoring(self, interval=1.0):
        """Start background GPU monitoring thread"""
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                stats = self.get_gpu_stats()
                if stats and self.rank < len(stats):
                    print(f"[Rank {self.rank}] GPU Util: {stats[self.rank]['utilization']:.1f}% | "
                          f"Mem: {stats[self.rank]['memory_used']:.0f}/{stats[self.rank]['memory_total']:.0f}MB | "
                          f"Temp: {stats[self.rank]['temperature']:.0f}°C")
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")
    
    def get_summary_stats(self):
        """Calculate summary statistics"""
        if not self.metrics_history:
            return {}
        
        gpu_utils = [m['gpu']['utilization'] for m in self.metrics_history if m.get('gpu')]
        throughputs = [m['throughput'] for m in self.metrics_history if 'throughput' in m]
        
        summary = {
            'rank': self.rank,
            'total_batches': len(self.metrics_history),
            'avg_gpu_utilization': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
            'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
            'max_throughput': max(throughputs) if throughputs else 0,
        }
        
        return summary
    
    def close(self):
        """Cleanup and final save"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        self.save_metrics()
        
        # Print summary
        summary = self.get_summary_stats()
        print(f"\n[Rank {self.rank}] Performance Summary:")
        print(f"  Avg GPU Utilization: {summary.get('avg_gpu_utilization', 0):.1f}%")
        print(f"  Avg Throughput: {summary.get('avg_throughput', 0):.0f} img/s")
        print(f"  Max Throughput: {summary.get('max_throughput', 0):.0f} img/s")


class PerformanceProfiler:
    """Profile distributed training performance"""
    
    def __init__(self, world_size):
        self.world_size = world_size
        self.timings = {
            'forward': deque(maxlen=100),
            'backward': deque(maxlen=100),
            'optimizer': deque(maxlen=100),
            'data_loading': deque(maxlen=100)
        }
    
    def time_operation(self, name):
        """Context manager for timing operations"""
        class Timer:
            def __init__(self, profiler, op_name):
                self.profiler = profiler
                self.op_name = op_name
                self.start = None
            
            def __enter__(self):
                torch.cuda.synchronize()
                self.start = time.time()
                return self
            
            def __exit__(self, *args):
                torch.cuda.synchronize()
                elapsed = time.time() - self.start
                self.profiler.timings[self.op_name].append(elapsed)
        
        return Timer(self, name)
    
    def get_stats(self):
        """Get timing statistics"""
        stats = {}
        for name, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times)
                stats[name] = {
                    'avg_ms': avg_time * 1000,
                    'count': len(times)
                }
        
        # Calculate communication overhead estimate
        total_time = sum(s['avg_ms'] for s in stats.values())
        if total_time > 0:
            # Backward includes gradient sync
            comm_overhead = (stats.get('backward', {}).get('avg_ms', 0) - 
                           stats.get('forward', {}).get('avg_ms', 0)) / total_time * 100
            stats['communication_overhead_pct'] = max(0, comm_overhead)
        
        return stats
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("Performance Profile:")
        print("="*60)
        for name, timing in stats.items():
            if isinstance(timing, dict):
                print(f"  {name.capitalize()}: {timing['avg_ms']:.2f}ms")
            else:
                print(f"  {name.replace('_', ' ').title()}: {timing:.1f}%")
        print("="*60 + "\n")


def check_gpu_availability():
    """Check GPU availability and print info"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("\n" + "="*60)
        print("GPU Configuration:")
        print("="*60)
        
        for line in result.stdout.strip().split('\n'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 4:
                print(f"  GPU {parts[0]}: {parts[1]} ({parts[3]} memory)")
        
        print("="*60 + "\n")
        return True
    except Exception as e:
        print(f"Error: Could not detect GPUs: {e}")
        return False


if __name__ == '__main__':
    # Test monitoring
    check_gpu_availability()
    
    monitor = GPUMonitor(rank=0)
    stats = monitor.get_gpu_stats()
    
    if stats:
        print("\nCurrent GPU Stats:")
        for gpu in stats:
            print(f"  GPU {gpu['gpu_id']}: {gpu['utilization']}% util, "
                  f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f}MB, "
                  f"{gpu['temperature']}°C")
    else:
        print("Could not retrieve GPU stats")