
import torch
from typing import List, Optional
import math
from .utils import print_rank0
import torch.nn as nn


class BlockTable():
        def __init__(self, block_size:int , max_num_blocks:int, dtype, device):
            self.physical_block_values = torch.full((max_num_blocks,), -1, dtype=torch.int32, device=device)
            self.block_size = block_size
            self.num_valid_blocks = 0
        
        def get_last_filled(self):
            return self.filled[self.num_valid_blocks-1]
        
        def get_last_physical_block_value(self):
            return self.physical_block_values[self.num_valid_blocks-1]
class PagedKVCache(nn.Module):
    
            
    def __init__(self,  k_shape, v_shape, batch_size:int, block_size: int, num_heads: int, head_dim: int,max_sequence_length:int, dtype, device ):
        """
        Initialize the blockd KV caching system.

        Args:
            # num_blocks (int): Number of blocks in the cache.
            block_size (int): Number of tokens per block.
            num_heads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            device (str): Device for cache storage.
        """
        super().__init__()

        # self.num_blocks = num_blocks
        self.k_shape = k_shape
        self.v_shape = v_shape
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.num_blocks = batch_size * (max_sequence_length//block_size)
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_num_blocks_per_sequence = (max_sequence_length//block_size)
        
        k_head_dim = k_shape[-1]
        v_head_dim = v_shape[-1]
        self.register_buffer(
            "keys", torch.zeros((self.num_blocks, block_size, num_heads, k_head_dim), device=self.device, dtype=self.dtype), persistent=False
        )
        self.register_buffer(
            "values", torch.zeros((self.num_blocks, block_size, num_heads, v_head_dim), device=self.device, dtype=self.dtype), persistent=False
        )
      
        
        
        self.filled = torch.zeros(
            (self.num_blocks,), device = self.device, dtype=torch.int32
        )
        self.block_usage  = [None] * self.num_blocks
        self.block_tables : List[self.BlockTable] = []
        
    def get_max_k_cache_tokens(self)->int:
        return max(sum([self.filled[b_num] for b_num in bt.filled]) for bt in self.block_tables)

    
  
                
    def add_to_cache(self, k: torch.Tensor , v: torch.Tensor, token_counter:torch.Tensor = None, initial_prompt_lengths:torch.Tensor = None):
        B, T = k.size(0), k.size(1)
        tokens_to_save = T
        partial_indices, full_blocks, final_blocks = self.allocate_blocks_new(tokens_to_save, token_counter, initial_prompt_lengths)
        
        # Carrying out the vectorized partial_index filling section (noit done during prefill - only done during the decode  or generate process)
        
        valid_partial_mask = partial_indices != -1
        if valid_partial_mask.any():
            batch_indices = torch.arange(B, device = self.device)
            valid_partial_batch_indices = batch_indices[valid_partial_mask]
            valid_partial_block_indices = partial_indices[valid_partial_mask]
            
            
            num_filled_in_partial = token_counter[valid_partial_batch_indices] % self.block_size
            partial_tokens_available = self.block_size - num_filled_in_partial
            partial_tokens_to_write = torch.clamp(partial_tokens_available, max=tokens_to_save)
            
            
            indices = (valid_partial_block_indices, num_filled_in_partial)
            
            self.keys[valid_partial_block_indices, num_filled_in_partial, :, :] = k[valid_partial_batch_indices].squeeze(1)
            self.values[valid_partial_block_indices, num_filled_in_partial, :, :] = v[valid_partial_batch_indices].squeeze(1)
            
            self.filled[valid_partial_block_indices] = num_filled_in_partial + partial_tokens_to_write #Updating the global filled tensor that tracks how many filled in each block
            tokens_to_save = torch.full((B,), T, device=k.device, dtype = torch.int32)
            tokens_to_save[valid_partial_batch_indices]-=partial_tokens_to_write  
        
        # Carrying out the full block filling process
        if (token_counter is None):
            full_block_indices = full_blocks.flatten()
            tokens_to_fill = tokens_to_save-(tokens_to_save%self.block_size)
            k_reshaped = k[:,:tokens_to_fill, :, :].reshape(len(full_block_indices), self.block_size, k.size(2), k.size(3))
            v_reshaped = v[:,:tokens_to_fill, :, :].reshape(len(full_block_indices), self.block_size, k.size(2), k.size(3))
            
            self.keys[full_block_indices, :, :, :] = k_reshaped
            self.values[full_block_indices, :, :, :] = v_reshaped
            self.filled[full_block_indices] = self.block_size
        
            tokens_to_save -= self.block_size * full_blocks.size(1)
        # Now completing the final fill of the pending blocks left
        
        valid_final_mask = final_blocks != -1
        if valid_final_mask.any():
            batch_indices = torch.arange(B, device = self.device)
            valid_final_batch_indices = batch_indices[valid_final_mask]
            valid_final_block_indices = final_blocks[valid_final_mask]
            if isinstance(tokens_to_save, torch.Tensor): tokens_to_save = max(tokens_to_save[valid_final_mask])
            self.keys[valid_final_block_indices, 0:tokens_to_save, :, :] = k[valid_final_batch_indices, T-tokens_to_save:, :, :]
            self.values[valid_final_block_indices, 0:tokens_to_save, :, :] = v[valid_final_batch_indices, T-tokens_to_save:, :, :]
            self.filled[valid_final_block_indices] = initial_prompt_lengths[valid_final_mask] if initial_prompt_lengths is not None else tokens_to_save
            
       
        
                
    def allocate_blocks(self, tokens_to_save, token_counter: torch.Tensor = None, initial_prompt_lengths:torch.Tensor = None):
        
        B = self.batch_size
        final_blocks = torch.full((B,),-1, device = self.device)
        full_blocks = torch.full((B, self.block_size), -1, device = self.device)
        partial_indices = torch.full((B,), -1, device = self.device)
        final_blocks = []
        full_blocks = []
        partial_indices = []
        free_blocks = [i  for i, usage in enumerate(self.block_usage) if usage is None ]
        for batch_id in range(B):
            
            if batch_id >= len(self.block_tables):
                block_table = BlockTable( self.block_size, self.max_num_blocks_per_sequence, self.dtype, self.device)
                self.block_tables.append(block_table)
            else:
                block_table = self.block_tables[batch_id]
            
            
            if block_table.num_valid_blocks == 0:
                available_in_last_block = 0
                partial_index = -1
            else:
                rem = token_counter[batch_id] % self.block_size
                available_in_last_block = 0 if rem == 0 else self.block_size - rem # remainder overlapping to prevent modulus confusion for number of blocks available in the last filled partial block
                partial_index = block_table.get_last_physical_block_value() if available_in_last_block >0 else -1
            
            needed_tokens = max(0, tokens_to_save-available_in_last_block)
            num_blocks = (needed_tokens + self.block_size -1)//self.block_size
            
            
            if len(free_blocks) < num_blocks:
                num_evicted_blocks = num_blocks - len(free_blocks)
                # clearing the least recently used num_evicted_blocks based on block_usage
                sorted_blocks = sorted(enumerate(self.block_usage), key = lambda x: x[1])
                for index, usage in sorted_blocks[:num_evicted_blocks]:
                    if self.block_usage[index] is not None:
                        self.block_usage[index] = None
                        free_blocks.append(index)
                        num_evicted_blocks-=1
                        if num_evicted_blocks==0: break
                    
                    
            new_blocks = free_blocks[:num_blocks]
            free_blocks = free_blocks[num_blocks:]
            
            for nb in new_blocks:
                self.block_usage[nb] = torch.cuda.Event(enable_timing=True) 
            
            partial_indices.append(partial_index)
            full_blocks.append(new_blocks[:-1])
            final_blocks.append(new_blocks[-1] if len(new_blocks) >0 else -1)
            
            actual_num_blocks_needed = (initial_prompt_lengths[batch_id] +self.block_size-1)//self.block_size if initial_prompt_lengths is not None else num_blocks
            block_table.physical_block_values[block_table.num_valid_blocks: block_table.num_valid_blocks + actual_num_blocks_needed] = torch.tensor(new_blocks[:actual_num_blocks_needed], device = self.device)
            block_table.num_valid_blocks+= actual_num_blocks_needed
        
        return  torch.tensor(partial_indices, device = self.device, dtype=torch.int32), torch.tensor(full_blocks, device= self.device, dtype=torch.int32),torch.tensor(final_blocks, device= self.device, dtype=torch.int32),
    
    

    def reset(self):
        self.__init__(
            self.k_shape,
            self.v_shape,
            self.batch_size,
            self.block_size,
            self.num_heads,
            self.head_dim,
            self.max_sequence_length,
            self.dtype, 
            self.device
        )
        
       



def paged_sdpa(
    q:torch.Tensor , # (B, nh, t, hd)
    max_key_length:int ,
    paged_cache: PagedKVCache,
    attn_mask: torch.Tensor,
    is_causal:bool = False,
    # enable_gqa: bool = False,
    dropout_p = 0.0,
):
    
    
   
    B, nh_q, T, hd = q.shape
    nh_k = paged_cache.keys.size(2)


    k_all = torch.zeros((B, max_key_length, nh_k,  hd), dtype=q.dtype, device = paged_cache.device)
    v_all = torch.zeros((B, max_key_length, nh_k,  hd), dtype=q.dtype, device = paged_cache.device)

    for batch_id, block_table in enumerate(paged_cache.block_tables):
        
        
        offset = 0
        for block_id, physical_block_num in enumerate(block_table.physical_block_values[:block_table.num_valid_blocks]):
            
            
            filled = paged_cache.filled[physical_block_num]
            k_all[batch_id, offset:offset+filled] = paged_cache.keys[physical_block_num][:filled]
            v_all[batch_id, offset:offset+filled] = paged_cache.values[physical_block_num][:filled]

            offset += filled
        
    k_all = k_all.transpose(1,2).contiguous()
    v_all = v_all.transpose(1,2).contiguous()
    
    
    if is_causal:
        context = torch.nn.functional.scaled_dot_product_attention(
            q, k_all, v_all, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=True
        )
    else:
        context = torch.nn.functional.scaled_dot_product_attention(
            q, k_all, v_all, dropout_p=dropout_p, attn_mask=attn_mask, enable_gqa=True
        ) 
        

    return context

    
    
