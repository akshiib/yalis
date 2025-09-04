#!/bin/bash
# select_gpu_device wrapper script
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsi
export MPICH_GPU_SUPPORT_ENABLED=0
#export CUDA_VISIBLE_DEVICES=3,2,1,0
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_NET="IB"
export NCCL_OOB_NET_ENABLE=1
export NCCL_GRAPH_MIXING_SUPPORT=0 # This is very important for performance

export JOBID=${SLURM_JOB_ID}
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}

#export NVSHMEM_REMOTE_TRANSPORT=ib
export NVSHMEM_DEBUG=INFO


#exec nsys profile -o yalistrace -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node $*
exec $*
