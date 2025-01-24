
import torch
from typing import List, Optional
import math
from .utils import print_rank0

        
class PagedKVCache:
    class BlockTable:
        def __init__(self, block_size:int ):
            self.physical_block_values = []
            self.filled = []
            self.block_size = block_size
    def __init__(self,  batch_size:int, block_size: int, num_heads: int, head_dim: int,max_sequence_length:int, dtype, device = torch.device("cuda")):
        """
        Initialize the blockd KV caching system.

        Args:
            # num_blocks (int): Number of blocks in the cache.
            block_size (int): Number of tokens per block.
            num_heads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            device (str): Device for cache storage.
        """
        # self.num_blocks = num_blocks
        self.num_blocks = batch_size * (max_sequence_length//block_size)
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        # Initialize the cache and metadata
        self.keys = torch.zeros(
            (self.num_blocks, block_size, num_heads, head_dim), device= torch.device(self.device), dtype=self.dtype
                
        )
        self.values = torch.zeros_like(self.keys)
        self.block_usage = [None] * self.num_blocks  # Tracks usage of each block (None = free)
        
        self.block_tables : List[self.BlockTable] = []
        
    def get_max_k_cache_tokens(self):
        return max(sum(bt.filled) for bt in self.block_tables)

    
    def add_to_cache(self, k: torch.Tensor , v: torch.Tensor):
        
        for batch_id, k_batch in enumerate(k):
            v_batch = v[batch_id]
            tokens_to_save = k_batch.shape[0]
            if batch_id >= len(self.block_tables):
                bt = self.BlockTable( self.block_size)
                allocated_blocks, partial_index = self.allocate_blocks(tokens_to_save, bt)

                self.block_tables.append(bt)
            else:
                bt = self.block_tables[batch_id]
                allocated_blocks, partial_index = self.allocate_blocks(tokens_to_save, bt)
            
            # print(f"allocated these blocks {allocated_blocks} for batch_id {batch_id}")
                
            # Take care of the partial block that needs to be filled here
            if not partial_index == None:
                # print("doing partial")
                num_filled_in_partial = bt.filled[-1] 
                
                partial_tokens_available = self.block_size - num_filled_in_partial
                partial_tokens_to_write = min(partial_tokens_available, tokens_to_save)
                partial_block_to_write = bt.physical_block_values[-1]
                
                print(f"{partial_tokens_to_write}--{partial_block_to_write}")
                self.keys[partial_block_to_write, num_filled_in_partial: num_filled_in_partial + partial_tokens_to_write , :, :] = k_batch[0: partial_tokens_to_write, :, :]
                self.values[partial_block_to_write, num_filled_in_partial: num_filled_in_partial + partial_tokens_to_write , :, :] = v_batch[0: partial_tokens_to_write, :, :]
                
                bt.filled[-1] += partial_tokens_to_write  # Updating the filled section of the last block
                tokens_to_save -= partial_tokens_to_write

            else:
                # print("not doing partial")
                partial_tokens_available = 0
                partial_tokens_to_write = 0
            
            # Fill in the new blocks that have been allocated
            if len(allocated_blocks) == 0:
                # No new blocks needed; everything fit in partial block
                continue
            offset = partial_tokens_to_write

            for al_block in allocated_blocks[:-1]:
                # print("full_block filling")
                self.keys[al_block, 0: self.block_size , :, :] = k_batch[offset: offset + self.block_size,:,:]
                self.values[al_block, 0: self.block_size , :, :] = v_batch[offset: offset + self.block_size,:,:]
                offset+= self.block_size
                tokens_to_save-=self.block_size
                bt.physical_block_values.append(al_block)
                bt.filled.append(self.block_size)
            # print("final block filling")
            # print(f"________________________{allocated_blocks[-1]}")
            self.keys[allocated_blocks[-1], 0:tokens_to_save, :, : ] = k_batch[offset:offset+tokens_to_save, :, :]
            self.values[allocated_blocks[-1], 0:tokens_to_save, :, : ] = v_batch[offset:offset+tokens_to_save, :, :]
            bt.physical_block_values.append(allocated_blocks[-1])
            bt.filled.append(tokens_to_save)
                
    def allocate_blocks(self, tokens_to_save, block_table: BlockTable):
        
          
        free_blocks = [i  for i, usage in enumerate(self.block_usage) if usage is None ]
        # already_present = self.block_size if block_table.filled == [] else block_table.filled[-1]
        # num_blocks = (tokens_to_save + already_present - self.block_size)//self.block_size + 1
        if block_table.filled == []:
            available_in_last_block = 0
            partial_index = None
        else:
            available_in_last_block = self.block_size - block_table.filled[-1]
            partial_index = len(block_table.filled)-1 if available_in_last_block >0 else None
        
        needed_tokens = max(0, tokens_to_save-available_in_last_block)
        num_blocks = (needed_tokens + self.block_size -1)//self.block_size
        
        # If the number of free blocks is less than the number of new blocks needed
        # We do the Eviction process
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
        
        
        final_blocks = free_blocks[:num_blocks] 
        for nb in new_blocks:
            self.block_usage[nb] = torch.cuda.Event(enable_timing=True) # This is for the LRU eviction method
        
        return final_blocks, partial_index
        
       


# Ensure that q and K and V are on the CUDA device
# Testing this is needed
# Enable gqa 
# Attention masking needs to be corrected
# Device coherence with removal of hard coded device is meeded
@torch.compile()
def paged_sdpa(
    q:torch.Tensor , # (B, t, nh, hd)
    max_key_length:int ,
    paged_cache: PagedKVCache,
    attn_mask: torch.Tensor,
    is_causal:bool = False,
    # enable_gqa: bool = False,
    dropout_p = 0.0,
):
    
    
    # First QK calcualtion
    B, nh_q, T, hd = q.shape
    nh_k = paged_cache.keys.size(2)
    attention_scores = torch.zeros(
        (B, nh_q,  T, max_key_length), device= torch.device("cuda"), dtype = q.dtype
    )
    attn_bias = torch.zeros(
        (B, nh_q, T, max_key_length), dtype=q.dtype, device =  torch.device("cuda")

    )
    if is_causal:
        temp_mask = torch.ones(T, max_key_length, dtype=torch.bool, device = torch.device("cuda")).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)
    else:    
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    scale_factor = 1 / math.sqrt(q.size(-1)) 
    for batch_id, block_table in enumerate(paged_cache.block_tables):
        offset = 0
        
        for block_id, physical_block_num in enumerate(block_table.physical_block_values):
            filled = block_table.filled[block_id] 
            # print(f"{filled} filled in block {physical_block_num}")
            # print_rank0(f"{batch_id} -- {block_id} -- {filled}")
            k_sub = paged_cache.keys[physical_block_num][:filled].repeat_interleave( nh_q// nh_k ,1).transpose(0,1)
            print(f"ksub shape {k_sub.shape}")
            # print_rank0("k_sub shape: " + k_sub.shape)
            # print_rank0("q[batch_id] shape: " + q[batch_id].shape)
            # print(f"k_sub {k_sub}")
            qk_sub = q[batch_id] @ k_sub.transpose(-2,-1) * scale_factor
            # print(qk_sub)
            attention_scores[batch_id, :,  :,offset: offset+filled] = qk_sub
            offset += filled
    # print(f"attention scores before bias {attention_scores}")
    # Now I need to consider the attention mask as well in this
    attention_scores += attn_bias
    attention_scores = torch.softmax(attention_scores, dim=-1)
    attention_scores = torch.dropout(attention_scores, dropout_p, train=False)
    
    print(f"attention weights after dropout shape {attention_scores.shape}")
    # Computing the multiplication with the final V matrix
    context = torch.zeros_like(q)
    
    for batch_id, block_table in enumerate(paged_cache.block_tables):
        offset = 0
        
        for block_id, physical_block_num in enumerate(block_table.physical_block_values):   
            print(physical_block_num)

            filled = block_table.filled[block_id]
            v_sub = paged_cache.values[physical_block_num][:filled].repeat_interleave( nh_q// nh_k ,1).transpose(0,1)
            context_sub = attention_scores[batch_id, :, :, offset :offset + filled ] @ v_sub 
            print(f"shape {context_sub.shape} -- {context_sub}")
            context[batch_id] += context_sub
            offset += filled
    
    
    return context

    
    
