import torch
import unittest
import torch.nn.functional as F
from yalis import PagedKVCache, paged_sdpa


class TestPagedSDPA(unittest.TestCase):

    def setUp(self):
        """Set up a small PagedKVCache and standard KV cache for testing."""
        self.batch_size = 8
        self.block_size = 64
        self.q_per_kv = 4 # Queries per key-value
        self.n_query_groups = 2 # Reduced num_heads in grouped-query attention
        self.head_dim = 128
        self.max_sequence_length = 256
        self.dtype = torch.float  # Ensure dtype consistency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize paged KV cache
        self.paged_cache = PagedKVCache(
            k_shape = (self.head_dim,),
            v_shape = (self.head_dim,),
            batch_size=self.batch_size,
            block_size=self.block_size,
            num_heads=self.n_query_groups,  # Using n_query_groups instead of full num_heads
            head_dim=self.head_dim,
            max_sequence_length=self.max_sequence_length,
            dtype=self.dtype,
            device=self.device
        )

        # Initialize a basic KV cache for torch.sdpa
        self.kv_cache = {
            "keys": torch.zeros((self.batch_size, self.max_sequence_length, self.n_query_groups, self.head_dim),
                                dtype=self.dtype, device=self.device),
            "values": torch.zeros((self.batch_size, self.max_sequence_length, self.n_query_groups, self.head_dim),
                                  dtype=self.dtype, device=self.device),
        }
        
    def build_mask_from_index(self, index, t_max):
        B = index.size(0)
        # Create a range [0, 1, 2, ..., t_max-1] and reshape to [1, t_max] so it can broadcast.
        arange_t = torch.arange(t_max, device=index.device).unsqueeze(0)
        # Compare to index[:, None]: [B, 1] which will broadcast to [B, t_max]
        return arange_t <= index.unsqueeze(1)


    def test_paged_sdpa_with_attn_mask_using_new_tokens(self):
        """Test paged_sdpa with an explicit attention mask, ensuring correct KV cache updates with new tokens."""
        
        tokens_to_store = 5  # Number of tokens already stored in cache
        new_tokens = 1       # New token to be added

        # Create random queries, keys, and values for past tokens
        q = torch.randn((self.batch_size, self.q_per_kv * self.n_query_groups, 1, self.head_dim),
                        dtype=self.dtype, device=self.device).contiguous()
        k_past = torch.randn((self.batch_size, tokens_to_store, self.n_query_groups, self.head_dim),
                            dtype=self.dtype, device=self.device).contiguous()
        v_past = torch.randn_like(k_past).contiguous()

        # **Add past tokens to the paged KV cache**
        self.paged_cache.add_to_cache(k_past, v_past)

        # **Manually update PyTorch KV Cache for past tokens**
        self.kv_cache["keys"][:, :tokens_to_store, :, :] = k_past
        self.kv_cache["values"][:, :tokens_to_store, :, :] = v_past

        # **Generate a new key-value pair for the latest token**
        k_new = torch.randn((self.batch_size, 1, self.n_query_groups, self.head_dim), 
                            dtype=self.dtype, device=self.device)
        v_new = torch.randn_like(k_new)

        # **Simulate appending the new token to the KV Cache**
        token_counter = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        token_counter+= tokens_to_store
        B = self.kv_cache["keys"].size(0)
        b_indices = torch.arange(B, device=q.device)

        # **Update KV Cache with new token**
        self.kv_cache["keys"][b_indices, token_counter.view(-1), :,:] = k_new[:, 0,:, :]
        self.kv_cache["values"][b_indices, token_counter.view(-1), :,:] = v_new[:, 0,:, :]

        # **Also add new token to paged KV cache**
        self.paged_cache.add_to_cache(k_new, v_new)

        # **Build attention mask (masks future tokens)**
        attn_mask_torch = self.build_mask_from_index(token_counter, t_max= self.kv_cache["keys"].size(1) )[:, None, None, :]
        max_key_length = self.paged_cache.get_max_k_cache_tokens()
        mask = self.build_mask_from_index(token_counter, t_max=max_key_length)[
                :, None, None, :
        ]
        # **Run paged SDPA**
        paged_sdpa_output = paged_sdpa(q, max_key_length, self.paged_cache, attn_mask = mask , is_causal=False, dropout_p=0.0)

        # Ensure dtype consistency
        paged_sdpa_output = paged_sdpa_output.to(self.dtype)

        # **Reshape queries for PyTorch SDPA**
        # q_reshaped = q.reshape(self.batch_size, self.q_per_kv * self.n_query_groups, tokens_to_store + new_tokens, self.head_dim)

        # **Run PyTorch SDPA with updated KV Cache**
        keys_shape = self.kv_cache["keys"][:, :tokens_to_store + new_tokens, :, :].transpose(1,2).shape
        reference_sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            self.kv_cache["keys"].transpose(1,2),
            self.kv_cache["values"].transpose(1,2),
            attn_mask=attn_mask_torch,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True
        )

        # Ensure dtype consistency
        print(f"reference_sdpa_output ->>>> {reference_sdpa_output.shape}")
        
        reference_sdpa_output = reference_sdpa_output.to(self.dtype).reshape(self.batch_size, self.q_per_kv * self.n_query_groups, new_tokens, self.head_dim)

        # Validate dtype consistency
        self.assertEqual(paged_sdpa_output.dtype, reference_sdpa_output.dtype, 
                        f"Mismatch in dtype: Paged SDPA ({paged_sdpa_output.dtype}) vs Reference SDPA ({reference_sdpa_output.dtype})")

        # Validate numerical correctness
        self.assertTrue(torch.allclose(paged_sdpa_output, reference_sdpa_output, atol=1e-3),
                        "Paged SDPA output does not match Reference SDPA output with explicit attention mask")

    def test_paged_sdpa_with_causal_mask(self):
        """Test paged_sdpa with a causal mask to ensure correct masking behavior."""
        tokens_to_store = 41

        # Create random queries, keys, and values
        q = torch.randn((self.batch_size, self.q_per_kv*self.n_query_groups, tokens_to_store, self.head_dim),
                        dtype=self.dtype, device=self.device).contiguous()
        k = torch.randn((self.batch_size, tokens_to_store, self.n_query_groups, self.head_dim),
                        dtype=self.dtype, device=self.device).contiguous()
        v = torch.randn_like(k).contiguous()

        # Add KV pairs to paged KV cache
        self.paged_cache.add_to_cache(k, v)
        
        # print(self.paged_cache.block_tables[0].filled)
        
        # self.paged_cache.keys[0][:tokens_to_store] = self.kv_cache["keys"][0][:tokens_to_store]
        # self.paged_cache.values[0][:tokens_to_store] = self.kv_cache["values"][0][:tokens_to_store]
        # Store KV pairs in standard PyTorch KV cache
        self.kv_cache["keys"][:, :tokens_to_store, :, :] = k
        self.kv_cache["values"][:, :tokens_to_store, :, :] = v

        self.assertEqual(self.paged_cache.keys[0][:tokens_to_store].shape, self.kv_cache["keys"][0][:tokens_to_store].shape, 
                        "Cache shapes does not match")
        self.assertTrue(torch.allclose(self.paged_cache.keys[0][:tokens_to_store], self.kv_cache["keys"][0][:tokens_to_store], atol=1e-3),
                        "Cache does not match")
        self.assertEqual(self.paged_cache.values[0][:tokens_to_store].shape, self.kv_cache["values"][0][:tokens_to_store].shape, 
                        "Cache shapes does not match")
        self.assertTrue(torch.allclose(self.paged_cache.values[0][:tokens_to_store], self.kv_cache["values"][0][:tokens_to_store], atol=1e-3),
                        "Cache does not match")
        # Run paged SDPA with causal masking
        paged_sdpa_output = paged_sdpa(q, tokens_to_store, self.paged_cache, None, is_causal=True, dropout_p=0.0)

        # Ensure dtype consistency
        paged_sdpa_output = paged_sdpa_output.to(self.dtype)

        # Reshape q for PyTorch SDPA (needs heads in dim 2)
        q_reshaped = q.reshape(self.batch_size, self.q_per_kv * self.n_query_groups   , tokens_to_store, self.head_dim)

        # Run PyTorch's built-in SDPA with causal masking
        torch_sdpa_output = F.scaled_dot_product_attention(
            q_reshaped,
            # self.kv_cache["keys"][:, :tokens_to_store, :, :].reshape(
            #     self.batch_size, self.n_query_groups, tokens_to_store, self.head_dim
            # ),
            # self.kv_cache["values"][:, :tokens_to_store, :, :].reshape(
            #     self.batch_size, self.n_query_groups, tokens_to_store, self.head_dim
            # ),
            k.transpose(1,2),
            v.transpose(1,2),
            dropout_p=0.0,
            is_causal=True,
            enable_gqa = True
        )#.reshape(self.batch_size, self.q_per_kv, self.n_query_groups, tokens_to_store, self.head_dim)

        # print(f"torch_sdpa shae {torch_sdpa_output.shape}")
        # Ensure dtype consistency
        torch_sdpa_output = torch_sdpa_output.to(self.dtype).reshape(self.batch_size, self.q_per_kv*self.n_query_groups, tokens_to_store, self.head_dim)

        # print(f"paged_sdpa shae {paged_sdpa_output}")
        # print(f"torch_sdpa shae {torch_sdpa_output}")
        # Validate dtype consistency
        self.assertEqual(paged_sdpa_output.dtype, torch_sdpa_output.dtype, 
                         f"Mismatch in dtype: Paged SDPA ({paged_sdpa_output.dtype}) vs Torch SDPA ({torch_sdpa_output.dtype})")

        # Validate numerical correctness
        self.assertTrue(torch.allclose(paged_sdpa_output, torch_sdpa_output, atol=1e-3),
                        "Paged SDPA output does not match PyTorch SDPA output with causal mask")

if __name__ == "__main__":
    unittest.main()
