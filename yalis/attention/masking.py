import torch
from torch.nn.attention.flex_attention import create_block_mask


def flex_decode_mask(token_counter):
    def _inner_mask(b, h, q_idx, kv_idx):
        return (kv_idx <= (token_counter[b]))
    return _inner_mask

def flex_prefill_mask(b, h, q_idx, kv_idx):
    return (kv_idx <= q_idx)
    
def flex_paged_decode_mask(paged_kv_cache_manager):
    def _inner_mask(b, h, q_idx, kv_idx):
        # kv_idx -> physical page -> if it belongs to 
        physical_page_id = (kv_idx // paged_kv_cache_manager.page_block_size())
        owner_of_page = paged_kv_cache_manager.page_to_sequence()[physical_page_id]
        return (owner_of_page == b)
    return _inner_mask

def create_causal_block_mask_for_flex_attention(Q_LEN, KV_LEN, paged_kv_cache_manager=None, token_counter=None):
    assert (token_counter is None) ^ (paged_kv_cache_manager is None), "Exactly one paged_kv_cache_manager or token_counter should be done"
    if Q_LEN > 1: # prefill - this is buggy on perlmutter
        #return create_block_mask(flex_prefill_mask, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN)
        return None
    # decode
    if paged_kv_cache_manager is None:
        return create_block_mask(flex_decode_mask(token_counter), 
                                 B=None, H=None, 
                                 Q_LEN=Q_LEN, KV_LEN=KV_LEN)
    else:
        return create_block_mask(flex_paged_decode_mask(paged_kv_cache_manager), 
                                 B=None, H=None, 
                                 Q_LEN=Q_LEN, KV_LEN=KV_LEN, BLOCK_SIZE=[1, paged_kv_cache_manager.page_block_size()])
        
