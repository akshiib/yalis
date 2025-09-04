#!/usr/bin/env python3
"""
Benchmark script to compare latencies of ring vs recursive allreduce algorithms.
Run with: mpirun -np 4 python benchmark_ring_vs_recursive.py
"""

import sys
import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# Add the build directory to the Python path so we can import the extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

try:
    from mpi4py import MPI
    print("✓ Successfully imported MPI4Py")
except ImportError as e:
    print(f"✗ Failed to import MPI4Py: {e}")
    print("Please install MPI4Py: pip install mpi4py")
    sys.exit(1)

try:
    import nvshmem_comm_cuda
    print("✓ Successfully imported nvshmem_comm_cuda extension")
except ImportError as e:
    print(f"✗ Failed to import nvshmem_comm_cuda extension: {e}")
    sys.exit(1)

class AllReduceBenchmark:
    def __init__(self, comm_wrapper, rank, world_size):
        self.comm_wrapper = comm_wrapper
        self.rank = rank
        self.world_size = world_size
        self.results = defaultdict(list)
        
        # Create CUDA stream for timing
        self.stream = torch.cuda.Stream()
        self.stream_ptr = self.stream.cuda_stream
        
        # Warm up GPU
        self._warmup()
    
    def _warmup(self):
        """Warm up the GPU to ensure consistent timing."""
        if self.rank == 0:
            print("Warming up GPU...")
        
        # Create a small tensor and run a few iterations
        tensor = torch.ones(1024, dtype=torch.int32, device=f"cuda:{self.rank % 4}")
        for _ in range(10):
            self.comm_wrapper.allreduce(tensor, self.stream_ptr, "ring")
            self.comm_wrapper.allreduce(tensor, self.stream_ptr, "recursive")
        
        torch.cuda.synchronize()
    
    def benchmark_single_size(self, tensor_size, dtype=torch.int32, num_iterations=100, warmup_iterations=10):
        """Benchmark a single tensor size for both algorithms."""
        local_rank = self.rank % 4
        tensor = torch.ones(tensor_size, dtype=dtype, device=f"cuda:{local_rank}")
        
        algorithms = ["ring", "recursive"]
        timings = {}
        
        for alg in algorithms:
            # Warmup
            for _ in range(warmup_iterations):
                self.comm_wrapper.allreduce(tensor, self.stream_ptr, alg)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(num_iterations):
                self.comm_wrapper.allreduce(tensor, self.stream_ptr, alg)
            end_event.record()
            
            torch.cuda.synchronize()
            
            # Calculate timing
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            avg_time = elapsed_time / num_iterations
            
            timings[alg] = {
                'avg_time_ms': avg_time,
                'total_time_ms': elapsed_time,
                'iterations': num_iterations
            }
            
            # Verify result
            expected_value = self.world_size
            if not torch.allclose(tensor, torch.ones(tensor_size, dtype=dtype, device=f"cuda:{local_rank}") * expected_value):
                print(f"✗ Verification failed for {alg} algorithm on rank {self.rank}")
                return None
        
        return timings
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across multiple tensor sizes."""
        if self.rank == 0:
            print("Starting comprehensive benchmark...")
            print("=" * 80)
        
        # Define tensor sizes to test (in elements)
        # Start small and scale up exponentially
        base_sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
        
        # For larger world sizes, we might want to test even larger tensors
        if self.world_size >= 8:
            base_sizes.extend([67108864, 268435456])
        
        # Test different data types
        dtypes = [torch.int32, torch.float32]
        
        all_results = {}
        
        for dtype in dtypes:
            dtype_name = str(dtype).split('.')[-1]
            if self.rank == 0:
                print(f"\nTesting {dtype_name} tensors...")
                print("-" * 60)
            
            dtype_results = {}
            
            for size in base_sizes:
                if self.rank == 0:
                    print(f"Testing size: {size:,} elements ({size * dtype().element_size() / 1024 / 1024:.2f} MB)")
                
                # Adjust iterations based on size to keep benchmark time reasonable
                if size <= 65536:
                    num_iterations = 200
                elif size <= 1048576:
                    num_iterations = 100
                elif size <= 16777216:
                    num_iterations = 50
                else:
                    num_iterations = 20
                
                timings = self.benchmark_single_size(size, dtype, num_iterations)
                
                if timings:
                    dtype_results[size] = timings
                    
                    if self.rank == 0:
                        ring_time = timings['ring']['avg_time_ms']
                        recursive_time = timings['recursive']['avg_time_ms']
                        speedup = recursive_time / ring_time if ring_time > 0 else 0
                        
                        print(f"  Ring: {ring_time:.4f} ms, Recursive: {recursive_time:.4f} ms")
                        print(f"  Speedup (Ring/Recursive): {speedup:.2f}x")
                        
                        # Determine which is faster
                        if ring_time < recursive_time:
                            print(f"  ✓ Ring is {recursive_time/ring_time:.2f}x faster")
                        elif recursive_time < ring_time:
                            print(f"  ✓ Recursive is {ring_time/recursive_time:.2f}x faster")
                        else:
                            print(f"  ✓ Both algorithms have similar performance")
                else:
                    if self.rank == 0:
                        print(f"  ✗ Benchmark failed for size {size}")
            
            all_results[dtype_name] = dtype_results
        
        return all_results
    
    def generate_plots(self, results, output_dir="benchmark_results"):
        """Generate performance comparison plots."""
        if self.rank != 0:
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        for dtype_name, dtype_results in results.items():
            if not dtype_results:
                continue
            
            sizes = list(dtype_results.keys())
            ring_times = [dtype_results[size]['ring']['avg_time_ms'] for size in sizes]
            recursive_times = [dtype_results[size]['recursive']['avg_time_ms'] for size in sizes]
            
            # Convert sizes to MB for x-axis
            sizes_mb = [size * 4 / 1024 / 1024 for size in sizes]  # Assuming 4 bytes per element
            
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Latency comparison
            plt.subplot(2, 2, 1)
            plt.loglog(sizes_mb, ring_times, 'o-', label='Ring', linewidth=2, markersize=8)
            plt.loglog(sizes_mb, recursive_times, 's-', label='Recursive', linewidth=2, markersize=8)
            plt.xlabel('Tensor Size (MB)')
            plt.ylabel('Latency (ms)')
            plt.title(f'AllReduce Latency Comparison - {dtype_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Speedup
            plt.subplot(2, 2, 2)
            speedups = [r/rr if rr > 0 else 0 for r, rr in zip(ring_times, recursive_times)]
            plt.semilogx(sizes_mb, speedups, 'o-', color='green', linewidth=2, markersize=8)
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
            plt.xlabel('Tensor Size (MB)')
            plt.ylabel('Speedup (Ring/Recursive)')
            plt.title('Performance Speedup')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Absolute difference
            plt.subplot(2, 2, 3)
            differences = [abs(r - rr) for r, rr in zip(ring_times, recursive_times)]
            plt.semilogx(sizes_mb, differences, 'o-', color='orange', linewidth=2, markersize=8)
            plt.xlabel('Tensor Size (MB)')
            plt.ylabel('Absolute Difference (ms)')
            plt.title('Absolute Performance Difference')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Percentage difference
            plt.subplot(2, 2, 4)
            pct_differences = [abs(r - rr) / max(r, rr) * 100 for r, rr in zip(ring_times, recursive_times)]
            plt.semilogx(sizes_mb, pct_differences, 'o-', color='purple', linewidth=2, markersize=8)
            plt.xlabel('Tensor Size (MB)')
            plt.ylabel('Percentage Difference (%)')
            plt.title('Percentage Performance Difference')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/benchmark_comparison_{dtype_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Generated plots for {dtype_name} in {output_dir}/")
    
    def save_results(self, results, output_dir="benchmark_results"):
        """Save benchmark results to JSON file."""
        if self.rank != 0:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Add metadata
        output_data = {
            'metadata': {
                'world_size': self.world_size,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'gpu_info': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown',
                'cuda_version': torch.version.cuda
            },
            'results': results
        }
        
        output_file = f'{output_dir}/benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved results to {output_file}")
    
    def print_summary(self, results):
        """Print a summary of the benchmark results."""
        if self.rank != 0:
            return
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        for dtype_name, dtype_results in results.items():
            if not dtype_results:
                continue
            
            print(f"\n{dtype_name.upper()} Results:")
            print("-" * 40)
            
            ring_wins = 0
            recursive_wins = 0
            ties = 0
            
            for size, timings in dtype_results.items():
                ring_time = timings['ring']['avg_time_ms']
                recursive_time = timings['recursive']['avg_time_ms']
                
                if ring_time < recursive_time:
                    ring_wins += 1
                elif recursive_time < ring_time:
                    recursive_wins += 1
                else:
                    ties += 1
            
            total_tests = len(dtype_results)
            print(f"Total tests: {total_tests}")
            print(f"Ring wins: {ring_wins} ({ring_wins/total_tests*100:.1f}%)")
            print(f"Recursive wins: {recursive_wins} ({recursive_wins/total_tests*100:.1f}%)")
            print(f"Ties: {ties} ({ties/total_tests*100:.1f}%)")
            
            # Find best and worst cases
            best_ring_speedup = 0
            best_recursive_speedup = 0
            worst_ring_speedup = float('inf')
            worst_recursive_speedup = float('inf')
            
            for size, timings in dtype_results.items():
                ring_time = timings['ring']['avg_time_ms']
                recursive_time = timings['recursive']['avg_time_ms']
                
                if recursive_time > 0:
                    ring_speedup = recursive_time / ring_time
                    best_ring_speedup = max(best_ring_speedup, ring_speedup)
                    worst_ring_speedup = min(worst_ring_speedup, ring_speedup)
                
                if ring_time > 0:
                    recursive_speedup = ring_time / recursive_time
                    best_recursive_speedup = max(best_recursive_speedup, recursive_speedup)
                    worst_recursive_speedup = min(worst_recursive_speedup, recursive_speedup)
            
            print(f"\nBest Ring speedup: {best_ring_speedup:.2f}x")
            print(f"Worst Ring speedup: {worst_ring_speedup:.2f}x")
            print(f"Best Recursive speedup: {best_recursive_speedup:.2f}x")
            print(f"Worst Recursive speedup: {worst_recursive_speedup:.2f}x")

def main():
    """Main benchmark function."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    if rank == 0:
        print("AllReduce Algorithm Benchmark: Ring vs Recursive")
        print(f"Running with {world_size} MPI processes")
        print("=" * 80)
    
    # Synchronize all processes
    comm.Barrier()
    
    try:
        # Create NVSHMEMCommWrapper instance
        comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, world_size, 0)
        
        # Set kernel parameters for consistent benchmarking
        comm_wrapper.set_kernel_params(32, 512, 262144)
        
        if rank == 0:
            print("✓ Successfully created NVSHMEMCommWrapper")
            print(f"✓ Kernel parameters: 32 blocks, 512 threads/block, 262144 chunk size")
        
        # Create benchmark instance
        benchmark = AllReduceBenchmark(comm_wrapper, rank, world_size)
        
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Generate plots and save results
        benchmark.generate_plots(results)
        benchmark.save_results(results)
        benchmark.print_summary(results)
        
        if rank == 0:
            print("\n🎉 Benchmark completed successfully!")
            print("Check the 'benchmark_results' directory for detailed results and plots.")
        
    except Exception as e:
        print(f"✗ Error during benchmarking on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Synchronize before finalizing
    comm.Barrier()
    MPI.Finalize()
    return True

if __name__ == "__main__":
    main() 