#include "nvshmem_comm.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <tuple>

// Recursive-doubling all-reduce using NVSHMEM.
// Templated over data type T.
template <typename T>
__global__ void recursive_allreduce_kernel_t(T *dst, T *src, size_t nreduce,
                                             uint64_t *signal, size_t chunk_size_bytes) {
    int mype  = nvshmem_my_pe();
    int npes  = nvshmem_n_pes();

    // Compute number of doubling steps = ceil(log2(npes))
    int steps = 0;
    while ((1u << steps) < (unsigned)npes) ++steps;

    // Block‐wise partitioning (same as before)
    int  block_idx    = blockIdx.x;
    int  thread_id    = threadIdx.x;
    int  num_threads  = blockDim.x;
    int  num_blocks   = gridDim.x;
    const size_t elems_per_block   = nreduce / num_blocks;
    if (elems_per_block * (block_idx) >= nreduce) return;


    // Slide dst/src pointers to this block’s slice
    dst = dst + block_idx * elems_per_block;
    src = src + block_idx * elems_per_block;       // scratch lives here
    signal = signal + block_idx * steps;                  // one signal word per block

    // Chunking inside each step
    size_t chunk_elems = max((size_t)32, chunk_size_bytes / sizeof(T));
    size_t num_chunks  = (elems_per_block + chunk_elems - 1) / chunk_elems;

    int partner;
    // --- recursive-doubling reduce ---
    for (int step = 2; step < steps ; ++step) {
        partner = mype ^ (1 << step);


        // Inform my partner that I am ready to receive their data and wait for them to do the same
        // This is important to ensure that if I have completed my partial sum from the previous partner, 
        // my partner might not have completed their partial sum from the previous partner, and I would
        // overwrite their data with my data.
        //if (thread_id == 0) {
        //    // Signal my partner that I am ready to receive their data
        //    nvshmem_uint64_atomic_set(signal + step, 1, partner);

        //    // Wait for my partner to signal me that they are ready to receive my data
        //    nvshmem_signal_wait_until(signal + step, NVSHMEM_CMP_GE, 1);
        //}
        //__syncthreads();
        // exchange & reduce each chunk
        for (size_t chunk_idx = 0; chunk_idx < num_chunks ; chunk_idx++) {
            size_t chunk_size = chunk_elems;
            T *my_chunk = src + chunk_idx * chunk_elems;
            T *scratch_chunk = (T*)dst + step * nreduce + chunk_idx * chunk_elems;

            // send my partial sum → partner’s scratch
            // if (thread_id < 32) {
            //     nvshmemx_putmem_signal_nbi_warp(
            //         (void*)scratch_chunk,
            //         (const void*)my_chunk,
            //         chunk_size * sizeof(T),
            //         signal + step,
            //         1,
            //         NVSHMEM_SIGNAL_ADD,
            //         partner);
            // }
            nvshmemx_putmem_signal_nbi_block(
                (void*)scratch_chunk,
                (const void*)my_chunk,
                chunk_size * sizeof(T),
                signal + step,
                1,
                NVSHMEM_SIGNAL_ADD,
                partner);

            // wait for partner’s data to arrive
            if (thread_id == 0) {
            	  //nvshmemx_signal_op(signal + step, 1, NVSHMEM_SIGNAL_ADD, partner);

                // The +1 is to account for the fact that the signal starts at 1, not 0
                nvshmem_signal_wait_until(signal + step, NVSHMEM_CMP_GE, chunk_idx + 1);
                //printf("[%d, %d] step: %d, chunk_idx: %lu, signal: %lu\n", mype, block_idx, step, chunk_idx, *(signal + step));

                //if (block_idx == 0 && thread_id == 0) {
                //  printf("[%d, %d] step: %d, chunk_idx: %lu, signal: %lu\n", mype, block_idx, step, chunk_idx, *(signal + step));
                //}

                // This is required to ensure I have sent my data before overwriting it with the partner's data
                // Without this, it can happen that I add the partner's data to my data before I have sent my data
                // TODO: This is a hack, we should find a better way to do this
                // nvshmem_quiet();

            }
            __syncthreads();

            // accumulate: dst += scratch
            for (size_t i = thread_id; i < chunk_size; i += num_threads) {
                scratch_chunk[i] = add_op(my_chunk[i], scratch_chunk[i]);
            }
        }
        __syncthreads();

        // The source pointer should be updated to the next step to the current latest buffer
        src = (T*)dst + step * nreduce;

      	//nvshmem_fence();
        if (thread_id == 0) {
             signal[step] = 0;
             __threadfence_system();
        }
    }
    // Reset the signal at the end of the kernel 
    // This is safe because only the current block waits 
    // for the signal to be set at this location.
    // If the block has reached this point, it has waited for 
    // all the signals and the signal can be cleared now.
    //if (thread_id == 0) {
    //  for (int i = 0; i < steps; i++) {
    //      signal[i] = 0;
    //  }
    //}
    //// TODO: This is probably not needed
    //__syncthreads();
    //nvshmem_quiet();
}



// Ring allreduce kernel implementation (templated)
template <typename T>
__global__ void ring_allreduce_kernel_t(T *dst, const T *src, size_t nreduce, 
                                        uint64_t *signal, size_t chunk_size_bytes) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    int num_blocks = gridDim.x;
    int block_idx = blockIdx.x;
    size_t elems_per_block = nreduce / num_blocks;

    // Each CTA will work independently
    if (elems_per_block * (blockIdx.x + 1) > nreduce) return;
    src = src + block_idx * elems_per_block;
    dst = dst + block_idx * elems_per_block;
    nreduce = elems_per_block;
    signal = signal + block_idx;

    size_t chunk_elems = chunk_size_bytes / sizeof(T);
    size_t num_chunks = nreduce / chunk_elems;

    // Reduce phase
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        if (mype != 0) {
            if (thread_id == 0) nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, chunk + 1);

            __syncthreads();
            for (size_t i = thread_id; i < chunk_elems; i += num_threads) {
                dst[i] = add_op(dst[i], src[i]);
            }
            __syncthreads();
        }
        if (thread_id == 0) {
            const void *send_src = (const void *)((mype == 0) ? src : dst);
            nvshmem_putmem_signal_nbi((void*)dst, send_src, chunk_elems * sizeof(T), signal, 1, NVSHMEM_SIGNAL_ADD, peer);
        }
        src = src + chunk_elems;
        dst = dst + chunk_elems;
    }

    // Broadcast phase
    dst = dst - num_chunks * chunk_elems;
    if (thread_id == 0) {
        for (size_t chunk = 0; chunk < num_chunks; chunk++) {
            if (mype < npes - 1) {  // Last pe already has the final result
                nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE,
                                          (mype == 0) ? chunk + 1 : num_chunks + chunk + 1);
            }
            if (mype < npes - 2) {
                nvshmem_putmem_signal_nbi((void*)dst, (const void*)dst, chunk_elems * sizeof(T), signal, 1, NVSHMEM_SIGNAL_ADD, peer);
            }
            dst = dst + chunk_elems;
        }
        *signal = 0;  // reset for next iteration
    }
}

// NVSHMEMCommWrapper implementation
NVSHMEMCommWrapper::NVSHMEMCommWrapper(int rank, int world_size, int device) 
    : rank_(rank), world_size_(world_size), device_(device), initialized_(false), signal_(nullptr), next_id_(0) {
    
    // Initialize MPI if not already initialized
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }

    // Set device
    CUDA_CHECK(cudaSetDevice(device_));

    // Initialize NVSHMEM with MPI
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Get PE information
    mype_ = nvshmem_my_pe();
    npes_ = nvshmem_n_pes();
    mype_node_ = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    // Verify rank consistency
    if (mype_ != rank_ || npes_ != world_size_) {
        throw std::runtime_error("MPI rank/world_size mismatch with NVSHMEM PE info");
    }

    // Set default parameters
    this->set_kernel_params(32, 512, 262144);

    initialized_ = true;
    std::cout << "NVSHMEM initialized for PE " << mype_ << " on " << npes_ << " PEs" << std::endl;
}


// Unique ID-based initialization
NVSHMEMCommWrapper::NVSHMEMCommWrapper(int rank, int world_size, int device, const torch::Tensor& unique_id_tensor)
    : rank_(rank), world_size_(world_size), device_(device), initialized_(false), signal_(nullptr), next_id_(0) {
    CUDA_CHECK(cudaSetDevice(device_));

    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;

    if (unique_id_tensor.numel() != sizeof(nvshmemx_uniqueid_t)) {
        throw std::runtime_error("unique_id_tensor has wrong size for nvshmemx_uniqueid_t");
    }
    memcpy(&uid, unique_id_tensor.data_ptr(), sizeof(uid));

    nvshmemx_set_attr_uniqueid_args(rank, world_size, &uid, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    mype_ = nvshmem_my_pe();
    npes_ = nvshmem_n_pes();
    mype_node_ = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    if (mype_ != rank_ || npes_ != world_size_) {
        throw std::runtime_error("Rank/world_size mismatch with NVSHMEM PE info");
    }

    this->set_kernel_params(32, 512, 262144);
    initialized_ = true;
}

NVSHMEMCommWrapper::~NVSHMEMCommWrapper() {
    destroy();
}

void NVSHMEMCommWrapper::destroy() {
    if (initialized_) {
        nvshmem_barrier_all();
        if (signal_) {
            nvshmem_free(signal_);
        }
        for (auto [id, ptr] : allocated_tensors_) {
            nvshmem_free(ptr);
        }
        for (auto [id, ptr] : allocated_scratch_) {
            nvshmem_free(ptr);
        }
        allocated_tensors_.clear();
        allocated_scratch_.clear();
        nvshmem_barrier_all();
        nvshmem_finalize();
    }

}

std::tuple<torch::Tensor, uint64_t> NVSHMEMCommWrapper::allocate_tensor(size_t size, torch::Dtype dtype, torch::Device device) {
    void *ptr = nvshmem_malloc(size * torch::elementSize(dtype));
    if (!ptr) {
        throw std::runtime_error("Failed to allocate tensor memory");
    }

    int steps = 0;
    while ((1u << steps) < (unsigned)npes_) {
        steps++;
    }

    // Create scratch memory
    void *scratch = nvshmem_malloc(steps * size * torch::elementSize(dtype));
    if (!scratch) {
        throw std::runtime_error("Failed to allocate scratch memory");
    }

    uint64_t id = next_id_.fetch_add(1);
    allocated_tensors_[id] = ptr;
    allocated_scratch_[id] = scratch;

    auto tensor = torch::from_blob(ptr, {static_cast<long>(size)}, torch::dtype(dtype).device(device));
    return std::make_tuple(tensor, id);
}

void NVSHMEMCommWrapper::free_tensor(uint64_t id) {
    if (id >= allocated_tensors_.size()) {
        throw std::runtime_error("Invalid tensor ID");
    }
    nvshmem_free(allocated_tensors_[id]);
    nvshmem_free(allocated_scratch_[id]);
    allocated_tensors_.erase(id);
    allocated_scratch_.erase(id);
}


void NVSHMEMCommWrapper::allreduce(torch::Tensor& tensor, uint64_t stream_ptr, std::string alg) {
    if (!initialized_) {
        throw std::runtime_error("NVSHMEM not initialized");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  
    int steps = 0;
    while ((1u << steps) < (unsigned)npes_) {
        steps++;
    }

    // Get tensor info
    size_t numel = tensor.numel();
    size_t size_bytes = numel * tensor.element_size();
    
    // Allocate symmetric memory for source and destination
    void *src_sym = nvshmem_malloc(size_bytes);
    void *dst_sym = nvshmem_malloc(size_bytes);
    
    if (!src_sym || !dst_sym) {
        throw std::runtime_error("Failed to allocate symmetric memory");
    }

    // Copy input tensor to symmetric memory
    CUDA_CHECK(cudaMemcpyAsync(src_sym, tensor.data_ptr(), size_bytes, 
                              cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t signal_size = 0;
    if (alg == "ring") {
      signal_size = num_blocks_;
    } else if (alg == "recursive") {
      signal_size = num_blocks_ * steps;
    } else {
      throw std::runtime_error("Invalid algorithm");
    }

    uint64_t *signal = (uint64_t *)nvshmem_calloc(signal_size, sizeof(uint64_t));
    if (!signal) {
        throw std::runtime_error("Failed to allocate signal memory");
    }

    // Set up kernel launch parameters
    dim3 gridDim(num_blocks_), blockDim(threads_per_block_);
    void *args[] = {&dst_sym, &src_sym, &numel, &signal, &chunk_size_};

    nvshmemx_barrier_all_on_stream(stream);

    // Launch kernels with dtype-based dispatch
    auto st = tensor.scalar_type();
    if (alg == "ring") {
        if (st == c10::kFloat) {
            nvshmemx_collective_launch((const void *)ring_allreduce_kernel_t<float>, gridDim, blockDim, args, 0, stream);
        } else if (st == c10::kHalf) {
            nvshmemx_collective_launch((const void *)ring_allreduce_kernel_t<__half>, gridDim, blockDim, args, 0, stream);
        } else if (st == c10::kBFloat16) {
            nvshmemx_collective_launch((const void *)ring_allreduce_kernel_t<__nv_bfloat16>, gridDim, blockDim, args, 0, stream);
        } else if (st == c10::kInt) {
            nvshmemx_collective_launch((const void *)ring_allreduce_kernel_t<int>, gridDim, blockDim, args, 0, stream);
        } else {
            throw std::runtime_error("Unsupported dtype for ring allreduce");
        }
    } else if (alg == "recursive") {
        if (st == c10::kFloat) {
            nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<float>, gridDim, blockDim, args, 0, stream);
        } else if (st == c10::kHalf) {
            nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<__half>, gridDim, blockDim, args, 0, stream);
        } else if (st == c10::kBFloat16) {
            nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<__nv_bfloat16>, gridDim, blockDim, args, 0, stream);
        } else if (st == c10::kInt) {
            nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<int>, gridDim, blockDim, args, 0, stream);
        } else {
            throw std::runtime_error("Unsupported dtype for recursive allreduce");
        }
    } else {
        throw std::runtime_error("Invalid algorithm");
    }

    // Synchronize

    nvshmemx_barrier_all_on_stream(stream);
    
    // Copy result back to tensor
    CUDA_CHECK(cudaMemcpyAsync(tensor.data_ptr(), dst_sym, size_bytes, 
                              cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Clean up
    nvshmem_free(src_sym);
    nvshmem_free(dst_sym);
    nvshmem_free(signal);
}

void NVSHMEMCommWrapper::allreduce_preallocated(torch::Tensor& tensor, uint64_t id, uint64_t stream_ptr, std::string alg) {
  if (!initialized_) {
    throw std::runtime_error("NVSHMEM not initialized");
  }

  size_t numel = tensor.numel();
  size_t numel_reduced = numel * 0.25;

  size_t size_bytes = numel * tensor.element_size();
  size_t size_bytes_reduced = size_bytes * 0.25;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  size_t chunk_size_reduced = chunk_size_ * 0.25;

  void *src_sym = tensor.data_ptr();
  void *dst_sym = allocated_scratch_[id];

  dim3 gridDim(num_blocks_), blockDim(threads_per_block_);
  void *args[] = {&dst_sym, &src_sym, &numel_reduced, &signal_, &chunk_size_reduced};

  int steps = 0;
  while ((1u << steps) < (unsigned)npes_) {
      steps++;
  }


  // The kernel should reset the signal at the end, so we don't need to do it here
  // nvshmemx_barrier_all_on_stream(stream);

  // Synchronize
  // nvshmemx_barrier_all_on_stream(stream);

  auto st = tensor.scalar_type();
  //nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<__nv_bfloat16>, gridDim, blockDim, args, 0, stream);
  // nvshmemx_sync_all_on_stream(stream);
  //nvshmemx_bfloat16_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, (__nv_bfloat16*)dst_sym, (const __nv_bfloat16*)src_sym, numel, stream);
  if (alg == "ring") {
    if (st == c10::kFloat) {
      nvshmemx_collective_launch((const void *)ring_allreduce_kernel_t<float>, gridDim, blockDim, args, 0, stream);
    } else if (st == c10::kHalf) {
      nvshmemx_collective_launch((const void *)ring_allreduce_kernel_t<__half>, gridDim, blockDim, args, 0, stream);
    } else if (st == c10::kBFloat16) {
      nvshmemx_collective_launch((const void *)ring_allreduce_kernel_t<__nv_bfloat16>, gridDim, blockDim, args, 0, stream);
    } else if (st == c10::kInt) {
      nvshmemx_collective_launch((const void *)ring_allreduce_kernel_t<int>, gridDim, blockDim, args, 0, stream);
    } else {
      throw std::runtime_error("Unsupported dtype for ring allreduce");
    }
  } else if (alg == "recursive") {
    if (st == c10::kFloat) {
      nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<float>, gridDim, blockDim, args, 0, stream);
    } else if (st == c10::kHalf) {
      nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<__half>, gridDim, blockDim, args, 0, stream);
    } else if (st == c10::kBFloat16) {
      nvshmemx_bfloat16_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE, (__nv_bfloat16*)src_sym, (const __nv_bfloat16*)src_sym, numel_reduced, stream);
      nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<__nv_bfloat16>, gridDim, blockDim, args, 0, stream);
      nvshmemx_bfloat16_fcollect_on_stream(NVSHMEMX_TEAM_NODE, (__nv_bfloat16*)src_sym, (const __nv_bfloat16*)dst_sym + (steps - 1) * numel_reduced, numel_reduced, stream);
    } else if (st == c10::kInt) {
      nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<int>, gridDim, blockDim, args, 0, stream);
    } else {
      throw std::runtime_error("Unsupported dtype for recursive allreduce");
    }
  } else {
    throw std::runtime_error("Invalid algorithm");
  }
  nvshmemx_quiet_on_stream(stream);

  //nvshmemx_barrier_all_on_stream(stream);
  //CUDA_CHECK(cudaStreamSynchronize(stream));
  //printf("[%d] Reached the end of kernel on rank %d\n", rank_, rank_);

  // Copy result back to tensor
  //CUDA_CHECK(cudaMemcpyAsync(tensor.data_ptr(), (const void*)dst_sym + (steps - 1) * size_bytes_reduced, size_bytes_reduced, 
                            //cudaMemcpyDeviceToDevice, stream));
  //CUDA_CHECK(cudaMemcpyAsync(tensor.data_ptr(), (const void*)dst_sym, size_bytes, 
                             //cudaMemcpyDeviceToDevice, stream));
}

void NVSHMEMCommWrapper::set_kernel_params(int num_blocks, int threads_per_block, size_t chunk_size) {
    num_blocks_ = num_blocks;
    threads_per_block_ = threads_per_block;
    chunk_size_ = chunk_size;

    int steps = 0;
    while ((1u << steps) < (unsigned)npes_) {
        steps++;
    }

    nvshmem_barrier_all();

    if (signal_) {
        nvshmem_free(signal_);
    }

    this->signal_size_ = num_blocks_ * steps;
    signal_ = (uint64_t *)nvshmem_calloc(signal_size_, sizeof(uint64_t));
    if (!signal_) {
        throw std::runtime_error("Failed to allocate signal memory");
    }

    nvshmem_barrier_all();
}

torch::Tensor NVSHMEMCommWrapper::get_unique_id_bytes() {
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_get_uniqueid(&uid);

    auto uid_tensor = torch::empty({sizeof(uid)}, torch::dtype(torch::kInt8).device(torch::kCPU));
    std::memcpy((void*)uid_tensor.data_ptr(), (void*)&uid, sizeof(uid));
    return uid_tensor;
}
