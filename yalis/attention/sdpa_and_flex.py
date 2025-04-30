from .registry import register_attention
import torch
from axonn import axonn as ax 
import math 
from axonn.intra_layer.communication import Drop, Gather
import torch.distributed as dist
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention
from .update_kv_cache import update_static_kv_cache, update_paged_kv_cache

def build_mask_from_index(index, t_max):
        B = index.size(0)
        # Create a range [0, 1, 2, ..., t_max-1] and reshape to [1, t_max] so it can broadcast.
        arange_t = torch.arange(t_max, device=index.device).unsqueeze(0)
        # Compare to index[:, None]: [B, 1] which will broadcast to [B, t_max]
        return arange_t <= index.unsqueeze(1)

def index_into_rope_cache_gen(cache, index):
    # index - [B, T]
    assert index.dim() == 1, "this method is only for the generation phase"
    return torch.index_select(cache, 0, index.view(-1)).reshape(
        index.size(0), 1, -1
    )

def create_upper_mask(dim, device):
    mask = torch.triu(torch.ones(dim, dim, dtype=torch.bool, device=device), diagonal=1)
    mask = mask.to(torch.float32)
    mask.masked_fill_(mask.bool(), -float("inf"))
    return mask

def intra_head_sdpa(q, k, v, attn_mask, process_group, enable_gqa, parallel=True):
    mask = create_upper_mask(q.size(2), q.device) if attn_mask is None else attn_mask
    if enable_gqa:
        B, h, n_q, d = q.shape
        g = k.size(1)
        hpg = h // g
        q = q.view(B, g, hpg, n_q, d)
        B2, g2, n_k, d2 = k.shape
        k = k.view(B2, g2, 1, n_k, d2)
        B3, g3, n_v, d3 = v.shape
        v = v.view(B3, g3, 1, n_v, d3)
    if parallel:
        q = Drop.apply(q, process_group).contiguous()
        #k = Drop.apply(k, process_group).contiguous()
    scale = 1.0 / math.sqrt(d)
    if enable_gqa:
        q = q * scale
        S = torch.einsum("b g h n d, b g o d t -> b g h n t", q, k.mT).clone().contiguous()
    else:
        q = q * scale
        S = (q @ k.mT).clone().contiguous()
    if parallel:
        dist.all_reduce(S, op=dist.ReduceOp.SUM, group=process_group)
    S = S + mask
    A = torch.nn.functional.softmax(S, dim=-1, dtype=torch.float).to(dtype=q.dtype)
    O = A @ v
    if enable_gqa:
        O = O.view(B, g * hpg, n_q, -1)
    O = Gather.apply(O, process_group)
    return O


def decode_attention(
    q: torch.Tensor,  # B,nh,1,hs
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
    token_counter: torch.Tensor,  # B,1
    use_intra_head_parallelism: bool = False,
    use_flex: bool = False,
    flex_attention_block_mask = None,
) -> torch.Tensor:
    enable_gqa = q.size(1) != k_cache.size(1)
    if use_intra_head_parallelism:
        assert not use_flex, "Intra head parallelism is not supported with flex attention"
        mask_float = torch.zeros_like(mask, dtype=torch.float32)
        mask_float = mask_float.masked_fill(~mask, float("-inf"))
        mask_float = mask_float[:, None, None, :]
        mask_float = mask_float.unsqueeze(1)
        out = intra_head_sdpa(
            q, k_cache, v_cache, mask_float,
            ax.comm_handle.inner_intra_layer_parallel_group,
            enable_gqa, parallel=True
        )
        return out
    else:
        if use_flex:
            assert flex_attention_block_mask is not None, "flex attention requires a block mask" 
            out = flex_attention(q, k_cache, v_cache, enable_gqa=enable_gqa, block_mask=flex_attention_block_mask)
        else:
            mask = build_mask_from_index(token_counter, t_max=k_cache.size(-2))
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k_cache, v_cache, attn_mask=mask[:, None, None, :], enable_gqa=enable_gqa
            )
        return out

def prefill_attention(
    q: torch.Tensor,  # B,nh,T,hs
    k: torch.Tensor,  # B,nh,T,hs
    v: torch.Tensor,  # B,nh,T,hs
    use_intra_head_parallelism: bool = False,
    use_flex: bool = False,
    flex_attention_block_mask = None,
) -> torch.Tensor:        
    enable_gqa = q.size(1) != k.size(1)
    use_intra_head_parallelism = False # for now we have disabled intra head parallelism for prefill
    if use_intra_head_parallelism: 
        # dead code
        out = intra_head_sdpa(
            q, k, v, None,
            ax.comm_handle.inner_intra_layer_parallel_group,
            enable_gqa, parallel=True
        )
        return out
    else:
        use_flex = False 
        if use_flex:
            assert flex_attention_block_mask is not None, "flex attention requires a block mask" 
            out = flex_attention(q, k, v, enable_gqa=enable_gqa, block_mask=flex_attention_block_mask)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        return out

def apply_rope(q, k, cos, sin, cache_seqlens):
    T = q.shape[-2]
    if T == 1:
        cos = index_into_rope_cache_gen(cos, cache_seqlens)
        sin = index_into_rope_cache_gen(sin, cache_seqlens)
        if cos.dim() > 1:
            # batch dimensions must align
            # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
            # we count from back because all of apply_rope does
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
    else:
        cos, sin = cos[:T], sin[:T]
        # cos and sin are of shape (T, hs)
        # we want to add singleton dimensions - (1, 1, T, hs)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

    roped_tensors = []
    for x in [q, k]:
        head_size = x.size(-1)
        x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
        x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
        rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
        roped = (x * cos) + (rotated * sin)
        roped = roped.to(dtype=x.dtype)
        roped_tensors.append(roped)

    q_roped, k_roped = roped_tensors

    return q_roped, k_roped



def sdpa_and_flex_attention(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,
              block_table: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              use_intra_head_parallelism: bool = False,
              use_flex: bool = False,
              flex_attention_block_mask = None,
              ) -> torch.Tensor:
    if use_flex: 
        assert flex_attention_block_mask is not None, "flex attention requires a block mask" 

    # apply rotary embeddings 
    if rotary_cos is not None and rotary_sin is not None:
        q, k = apply_rope(q, k, rotary_cos, rotary_sin, cache_seqlens)

    # update kv-cache
    k_to_update = Drop.apply(k, ax.comm_handle.inner_intra_layer_parallel_group) if use_intra_head_parallelism else k
    v_to_update = Drop.apply(v, ax.comm_handle.inner_intra_layer_parallel_group) if use_intra_head_parallelism else v

    if block_table is None:
        update_static_kv_cache(k_to_update, v_to_update, k_cache, v_cache, cache_seqlens)
    else:
        # note that these transposes/permutes do not change the data layout. they just change the strides of the tensor.
        # the transposes will "actually" happen if you call .contiguous() on the tensors, which we do not do here.
        update_paged_kv_cache(k=k_to_update.transpose(1,2), # [B, H, S_k, D] -> [B, S_k, H, D] 
                              v=v_to_update.transpose(1,2), # [B, H, S_k, D] -> [B, S_k, H, D]
                              block_table=block_table,
                              cache_seq_len=cache_seqlens,
                              k_cache=k_cache.permute([1, 2, 0, 3]), # [H, NUM_PAGES, PAGE_BLOCK_SIZE, D] -> [NUM_PAGES, PAGE_BLOCK_SIZE, H, D]
                              v_cache=v_cache.permute([1, 2, 0, 3])) # [H, NUM_PAGES, PAGE_BLOCK_SIZE, D] -> [NUM_PAGES, PAGE_BLOCK_SIZE, H, D]

    T = q.shape[-2]
    if T==1:
        y = decode_attention(
                     q,
                     k_cache, 
                     v_cache,  
                     cache_seqlens,
                     use_intra_head_parallelism,
                     use_flex, 
                     flex_attention_block_mask,
                 )
    else:
        y = prefill_attention(q=q, 
                              k=k, 
                              v=v, 
                              use_intra_head_parallelism=use_intra_head_parallelism, 
                              use_flex=use_flex, 
                              flex_attention_block_mask=flex_attention_block_mask)       
    return y

@register_attention("sdpa")
def sdpa_attention(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              use_intra_head_parallelism: bool = False,
              ):
    return sdpa_and_flex_attention(q=q,
                                   k=k,
                                   v=v,
                                   k_cache=k_cache, 
                                   v_cache=v_cache,
                                   cache_seqlens=cache_seqlens,
                                   rotary_cos=rotary_cos,
                                   rotary_sin=rotary_sin,
                                   use_intra_head_parallelism=use_intra_head_parallelism,
                                   use_flex=False,
                                   )

@register_attention("flex")
def flex_attention_(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              use_intra_head_parallelism: bool = False,
              **kwargs):  
    #assert "flex_attention_block_mask" in kwargs, "flex attention requires a block mask"
    #flex_attention_block_mask = kwargs.pop("flex_attention_block_mask")
    return sdpa_and_flex_attention(q=q,
                            k=k,
                            v=v,
                            k_cache=k_cache, 
                            v_cache=v_cache,
                            cache_seqlens=cache_seqlens,
                            rotary_cos=rotary_cos,
                            rotary_sin=rotary_sin,
                            use_intra_head_parallelism=use_intra_head_parallelism,
                            use_flex=True,
                            **kwargs,
                            )
