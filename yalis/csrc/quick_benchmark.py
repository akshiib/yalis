#!/usr/bin/env python3
"""
Quick benchmark script to compare latencies of ring vs recursive allreduce algorithms vs PyTorch distributed.
This is a simplified version for faster testing.
Run with: mpirun -np 4 python quick_benchmark.py
"""

import sys
import os
import torch
import time
# Add the build directory to the Python path so we can import the extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

from nvshmem_comm import NVSHMEMCommunicator


try:
    import nvshmem_comm_cuda
    print("✓ Successfully imported nvshmem_comm_cuda extension")
except ImportError as e:
    print(f"✗ Failed to import nvshmem_comm_cuda extension: {e}")
    sys.exit(1)

def init_torch_distributed():
    """Initialize PyTorch distributed backend."""
    try:
        # Get environment variables for distributed training
        # Initialize PyTorch distributed
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )

        local_rank = torch.distributed.get_rank() % 4
        torch.cuda.set_device(local_rank)
        
        print(f"✓ Successfully initialized PyTorch distributed ({torch.distributed.get_rank()}/{torch.distributed.get_world_size()})")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize PyTorch distributed: {e}")
        return False

def quick_benchmark():
    """Quick benchmark comparing ring vs recursive vs PyTorch distributed algorithms."""

    # Initialize PyTorch distributed
    torch_dist_initialized = init_torch_distributed()
    if not torch_dist_initialized:
        print("Warning: PyTorch distributed not available, skipping that comparison")
        sys.exit(1)
    

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = rank % 4
    
    if rank == 0:
        print("Quick AllReduce Benchmark: Ring vs Recursive vs PyTorch Distributed")
        print(f"Running with {world_size} MPI processes")
        print("=" * 80)
    
    # Synchronize all processes
    torch.distributed.barrier()
    
    # Create NVSHMEMCommWrapper instance
    local_ranok = rank % 4
    if torch_dist_initialized:
        nvcomm = NVSHMEMCommunicator(torch.distributed.group.WORLD)
        comm_wrapper = nvcomm.core
    else:
        comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, world_size, local_rank)


    
    # Set kernel parameters
    comm_wrapper.set_kernel_params(32, 512, 262144)
    
    if rank == 0:
        print("✓ Successfully created NVSHMEMCommWrapper")
    
    # Create CUDA stream
    stream = torch.cuda.Stream()
    stream_ptr = stream.cuda_stream

    # CUDA Graphs toggle
    use_cuda_graphs = os.getenv("USE_CUDA_GRAPHS", "0").lower() in ["1", "true", "yes", "on"]
    graphs_target = os.getenv("CUDA_GRAPHS_TARGET", "both").lower()  # pytorch|nvshmem|both
    use_graphs_pytorch = use_cuda_graphs and graphs_target in ["pytorch", "both"]
    use_graphs_nvshmem = use_cuda_graphs and graphs_target in ["nvshmem", "both"]
    if rank == 0:
        if use_cuda_graphs:
            print(f"CUDA Graphs: ENABLED (target={graphs_target}; USE_CUDA_GRAPHS=1, CUDA_GRAPHS_TARGET=pytorch|nvshmem|both)")
        else:
            print("CUDA Graphs: disabled (set USE_CUDA_GRAPHS=1 to enable)")
    
    # Test sizes (in elements)
    test_sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
    #test_sizes = [1024 * 1024]
    
    # Warm up
    if rank == 0:
        print("Warming up...")
    
    local_rank = rank % 4
    warmup_tensor, warmup_tensor_id = comm_wrapper.allocate_tensor(1024, torch.bfloat16, torch.device(f"cuda:{local_rank}"))
    warmup_tensor.fill_(1)
    #warmup_tensor = torch.ones(1024, dtype=torch.int32, device=f"cuda:{local_rank}")
    comm_wrapper.set_kernel_params(32, 512, 32768) 
    for _ in range(5):
        comm_wrapper.allreduce_preallocated(warmup_tensor, warmup_tensor_id, stream_ptr, "ring")
        comm_wrapper.allreduce_preallocated(warmup_tensor, warmup_tensor_id, stream_ptr, "recursive")
        if torch_dist_initialized:
            torch.distributed.all_reduce(warmup_tensor)
    
    torch.cuda.synchronize()
    comm_wrapper.free_tensor(warmup_tensor_id)

    
    if rank == 0:
        print("Starting benchmark...")
        header = f"{'Size (MB)':<12} {'Ring (ms)':<12} {'Recursive (ms)':<15} {'PyTorch (ms)':<15} {'Best':<10}"
        print(header)
        print("-" * len(header))
    
    results = {}

    for size in test_sizes:
        # Keeping number of chunks per block = 4
        comm_wrapper.set_kernel_params(4, 256, size // 4)

        # Create tensor
        tensor, tensor_id = comm_wrapper.allocate_tensor(size, torch.bfloat16, torch.device(f"cuda:{local_rank}"))
        tensor.fill_(1)

        # Compute message size in MB based on actual tensor dtype
        size_mb = size * tensor.element_size() / 1024 / 1024
        
        # Test ring algorithm
        torch.cuda.synchronize()

        # Optionally capture CUDA Graphs for each algorithm at this size
        ring_graph = None
        recursive_graph = None
        pytorch_graph = None
        if use_cuda_graphs:
            # Ensure the capture stream is idle
            stream.synchronize()
            try:
                if use_graphs_nvshmem:
                    torch.distributed.barrier()
                    torch.cuda.synchronize()
                    ring_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(ring_graph, stream=stream):
                        with torch.cuda.stream(stream):
                            comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, "ring")
                    torch.cuda.synchronize()
                    if rank == 0:
                        print(f"Captured CUDA Graph for Ring, size={size}")
            except Exception as e:
                ring_graph = None
                if use_graphs_nvshmem and rank == 0:
                    print(f"Failed to capture CUDA Graph for Ring (size={size}): {e}")

            try:
                if use_graphs_nvshmem:
                    torch.distributed.barrier()
                    torch.cuda.synchronize()
                    recursive_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(recursive_graph, stream=stream):
                        with torch.cuda.stream(stream):
                            comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, "recursive")
                    torch.cuda.synchronize()
                    if rank == 0:
                        print(f"Captured CUDA Graph for Recursive, size={size}")
            except Exception as e:
                recursive_graph = None
                if use_graphs_nvshmem and rank == 0:
                    print(f"Failed to capture CUDA Graph for Recursive (size={size}): {e}")

            if torch_dist_initialized and use_graphs_pytorch:
                try:
                    torch.distributed.barrier()
                    torch.cuda.synchronize()
                    pytorch_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(pytorch_graph, stream=stream):
                        with torch.cuda.stream(stream):
                            torch.distributed.all_reduce(tensor)
                    torch.cuda.synchronize()
                    if rank == 0:
                        print(f"Captured CUDA Graph for PyTorch all_reduce, size={size}")
                except Exception as e:
                    pytorch_graph = None
                    if rank == 0 and use_graphs_pytorch:
                        print(f"Failed to capture CUDA Graph for PyTorch all_reduce (size={size}): {e}")
        

        #print(f"Running ring algorithm for size {size}")
        torch.distributed.barrier()

        ring_time = 0
        for _ in range(50):  # 20 iterations for averaging
            with torch.cuda.stream(stream):
                tensor.fill_(1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_cuda_graphs and ring_graph is not None:
                ring_graph.replay()
            else:
                with torch.cuda.stream(stream):
                    comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, "ring")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ring_time += (t1 - t0) * 1000.0

            time.sleep(0.01)
            torch.distributed.barrier()

        ring_time = ring_time / 50
        
        # Test recursive algorithm
        torch.distributed.barrier()
        
        recursive_time = 0
        for _ in range(50):  # 20 iterations for averaging
            with torch.cuda.stream(stream):
                tensor.fill_(1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_cuda_graphs and recursive_graph is not None:
                recursive_graph.replay()
            else:
                with torch.cuda.stream(stream):
                    comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, "recursive")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            recursive_time += (t1 - t0) * 1000.0

            time.sleep(0.01)
            torch.distributed.barrier()

        recursive_time = recursive_time / 50

        # Test PyTorch distributed algorithm
        pytorch_time = 0
        if torch_dist_initialized:
            torch.distributed.barrier()
            
            for _ in range(50):  # 20 iterations for averaging
                with torch.cuda.stream(stream):
                    tensor.fill_(1)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if use_cuda_graphs and pytorch_graph is not None:
                    pytorch_graph.replay()
                else:
                    # torch.distributed/NCCL may use its own stream; time with device sync
                    torch.distributed.all_reduce(tensor)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                pytorch_time += (t1 - t0) * 1000.0

                time.sleep(0.01)
                torch.distributed.barrier()

            pytorch_time = pytorch_time / 50
        else:
            pytorch_time = float('inf')  # Mark as unavailable
        
        # Find the fastest algorithm
        times = {'Ring': ring_time, 'Recursive': recursive_time}
        if torch_dist_initialized:
            times['PyTorch'] = pytorch_time
        
        fastest_algorithm = min(times, key=times.get)
        fastest_time = times[fastest_algorithm]
        
        # Calculate speedups relative to fastest
        speedups = {}
        for algo, time_val in times.items():
            if time_val > 0:
                speedups[algo] = time_val / fastest_time
            else:
                speedups[algo] = 1.0
        
        if rank == 0:
            if torch_dist_initialized:
                print(f"{size_mb:<12.2f} {ring_time:<12.4f} {recursive_time:<15.4f} {pytorch_time:<15.4f} {fastest_algorithm:<10}")
            else:
                print(f"{size_mb:<12.2f} {ring_time:<12.4f} {recursive_time:<15.4f} {'N/A':<15} {fastest_algorithm:<10}")
        
        results[size] = {
            'ring_time': ring_time,
            'recursive_time': recursive_time,
            'pytorch_time': pytorch_time if torch_dist_initialized else None,
            'fastest_algorithm': fastest_algorithm,
            'fastest_time': fastest_time,
            'speedups': speedups
        }

        torch.distributed.barrier()

    # Ensure all GPU work is done before destruction/finalization
    torch.cuda.synchronize()
    torch.distributed.barrier()
    comm_wrapper.destroy()

    # Print summary
    if rank == 0 and results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        # Count wins for each algorithm
        wins = {'Ring': 0, 'Recursive': 0, 'PyTorch': 0}
        for r in results.values():
            wins[r['fastest_algorithm']] += 1
        
        total_tests = len(results)
        print(f"Total tests: {total_tests}")
        print(f"Ring wins: {wins['Ring']} ({wins['Ring']/total_tests*100:.1f}%)")
        print(f"Recursive wins: {wins['Recursive']} ({wins['Recursive']/total_tests*100:.1f}%)")
        if torch_dist_initialized:
            print(f"PyTorch wins: {wins['PyTorch']} ({wins['PyTorch']/total_tests*100:.1f}%)")
        
        # Find best performances
        for algo in ['Ring', 'Recursive', 'PyTorch']:
            if algo in wins and wins[algo] > 0:
                best_speedup = max(r['speedups'][algo] for r in results.values() if algo in r['speedups'])
                print(f"Best {algo} speedup: {best_speedup:.2f}x")
        
        print("\n🎉 Quick benchmark completed!")
        
    # Synchronize before finalizing
    torch.distributed.barrier()
    
    # Clean up PyTorch distributed
    if torch_dist_initialized:
        torch.distributed.destroy_process_group()
    
    return True

if __name__ == "__main__":
    quick_benchmark() 
