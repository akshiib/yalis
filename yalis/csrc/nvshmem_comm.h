#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <cuda_fp16.h>
#include <atomic>

// Include NVSHMEM headers to get proper type definitions
#include <nvshmem.h>
#include <nvshmemx.h>


// CUDA error checking macro
#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            throw std::runtime_error(std::string("CUDA failed: ") + cudaGetErrorString(result)); \
        }                                                                         \
    } while (0)

template <typename T>
__device__ inline T add_op(T a, T b) { return a + b; }

template <>
__device__ inline __half add_op(__half a, __half b) { return __hadd(a, b); }

template <>
__device__ inline __nv_bfloat16 add_op(__nv_bfloat16 a, __nv_bfloat16 b) {
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    return __float2bfloat16(fa + fb);
}

// Generic signaling put using byte-size for portability across types
__device__ inline void put_signal_block_bytes(void *dst, const void *src, size_t bytes,
                                              uint64_t *signal, uint64_t val, int op, int pe) {
    nvshmemx_putmem_signal_block(dst, src, bytes, signal, val, op, pe);
}

// Recursive-doubling all-reduce using NVSHMEM (templated)
template <typename T>
__global__ extern void recursive_allreduce_kernel_t(T *dst, T *src, size_t nreduce,
                                             uint64_t *signal, size_t chunk_size_bytes);

// Ring all-reduce using NVSHMEM (templated)
template <typename T>
__global__ extern void ring_allreduce_kernel_t(T *dst, const T *src, size_t nreduce, 
                                     uint64_t *signal, size_t chunk_size);



class NVSHMEMCommWrapper {
public:
    NVSHMEMCommWrapper(int rank, int world_size, int device);
    // Initialize using NVSHMEM unique id based attributes; unique_id is the raw bytes returned by nvshmemx_get_uniqueid
    NVSHMEMCommWrapper(int rank, int world_size, int device, const torch::Tensor& unique_id_bytes);
    ~NVSHMEMCommWrapper();

    // Disable copy constructor and assignment operator
    NVSHMEMCommWrapper(const NVSHMEMCommWrapper&) = delete;
    NVSHMEMCommWrapper& operator=(const NVSHMEMCommWrapper&) = delete;

    void destroy();

    std::tuple<torch::Tensor, uint64_t> allocate_tensor(size_t size, torch::Dtype dtype, torch::Device device);
    void free_tensor(uint64_t id);

    // Main communication methods
    void allreduce(torch::Tensor& tensor, uint64_t stream_ptr, std::string alg = "ring");

    void allreduce_preallocated(torch::Tensor& tensor, uint64_t id, uint64_t stream_ptr, std::string alg = "ring");
    
    // Configuration methods
    void set_kernel_params(int num_blocks, int threads_per_block, size_t chunk_size);
    
    // Getter methods
    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }
    int get_mype() const { return mype_; }
    int get_npes() const { return npes_; }

    static torch::Tensor get_unique_id_bytes();

private:
    int rank_;
    int world_size_;
    int device_;
    int mype_;
    int npes_;
    int mype_node_;
    size_t signal_size_;
    bool initialized_;
    
    // Kernel parameters
    int num_blocks_;
    int threads_per_block_;
    size_t chunk_size_;

    nvshmemx_uniqueid_t uid_ = NVSHMEMX_UNIQUEID_INITIALIZER;

    // Memory Pools
    std::unordered_map<uint64_t, void *> allocated_tensors_;
    std::unordered_map<uint64_t, void *> allocated_scratch_;
    std::atomic<uint64_t> next_id_;
    void *signal_;
}; 