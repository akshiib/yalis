import triton
import triton.language as tl
import torch

@triton.jit
def update_paged_kv_cache_kernel(
    k_ptr, v_ptr,
    block_table_ptr,
    cache_seq_len_ptr,
    cache_k_ptr, cache_v_ptr,
    B, S, H, D,
    max_pages_per_seq,
    page_block_size,
    pt_block_stride, pt_token_stride, pt_head_stride, pt_feature_stride,
    key_batch_stride, key_token_stride, key_head_stride, key_feature_stride,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B * S:
        return

    # Token index
    b = pid // S
    s = pid % S

    # calculate physical block id 
    cache_offset = tl.load(cache_seq_len_ptr + b)
    token_offset = cache_offset + s
    logical_block_id = token_offset // page_block_size
    offset_in_block = token_offset % page_block_size
    physical_block_id = tl.load(block_table_ptr + b * max_pages_per_seq + logical_block_id)

    # Shared across layouts
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    for h_offset in range(0, H, BLOCK_H):
        offs_h = tl.arange(0, BLOCK_H) + h_offset
        mask_h = offs_h < H

        offs_h_broadcast = offs_h[:, None]
        offs_d_broadcast = offs_d[None, :]

        src_ptr_offsets = key_batch_stride * b + key_token_stride * s + offs_h_broadcast * key_head_stride + offs_d_broadcast * key_feature_stride
        k_src_ptrs = k_ptr + src_ptr_offsets 
        v_src_ptrs = v_ptr + src_ptr_offsets

        # Unified layout: [num_blocks, page_block_size, H, D]
        dst_base = physical_block_id * pt_block_stride + offset_in_block * pt_token_stride 
        dst_ptrs = dst_base + offs_h_broadcast * pt_head_stride + offs_d_broadcast * pt_feature_stride

        k_dst_ptrs = cache_k_ptr + dst_ptrs
        v_dst_ptrs = cache_v_ptr + dst_ptrs

        k_vals = tl.load(k_src_ptrs, mask=mask_h[:, None] & mask_d[None, :])
        v_vals = tl.load(v_src_ptrs, mask=mask_h[:, None] & mask_d[None, :])
        tl.store(k_dst_ptrs, k_vals, mask=mask_h[:, None] & mask_d[None, :])
        tl.store(v_dst_ptrs, v_vals, mask=mask_h[:, None] & mask_d[None, :])




def update_paged_kv_cache( k:torch.Tensor, 
                           v:torch.Tensor,
                           block_table:torch.Tensor, 
                           cache_seq_len:torch.Tensor,
                           k_cache:torch.Tensor, 
                           v_cache:torch.Tensor):
    # note that k and v are of shape [B, S, H, D]
    # and k_cache and v_cache are of shape [num_blocks, page_block_size, H, D]
    
    B, S, H, D = k.shape
    BLOCK_H = min(1024 // D, H)        

    grid = (B * S,)    # one program per token

    max_pages_per_seq = block_table.shape[1]
    page_block_size = k_cache.shape[1]

    update_paged_kv_cache_kernel[grid](
        k, v,
        block_table, cache_seq_len,
        k_cache, v_cache,
        B, S, H, D,
        max_pages_per_seq,
        page_block_size,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        BLOCK_H=BLOCK_H,
        BLOCK_D=D
    )

def update_static_kv_cache(k,
                           v,
                           k_cache,
                           v_cache,
                           cache_seqlens):
    # head-first layout
    B = k.size(0)
    T = k.size(2)
    if T == 1:
        b_indices = torch.arange(B, device=k_cache.device)
        t_indices = cache_seqlens.view(-1)
        
        k_cache[b_indices, :, t_indices, :] = k[:, :, 0, :]
        v_cache[b_indices, :, t_indices, :] = v[:, :, 0, :]
    else:
        k_cache[:, :, :T, :] = k[:, :, :T, :]
        v_cache[:, :, :T, :] = v[:, :, :T, :]