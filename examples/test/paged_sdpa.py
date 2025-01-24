import torch
import unittest
import torch.nn.functional as F
from yalis import PagedKVCache, paged_sdpa


class TestPagedSDPA(unittest.TestCase):

    def setUp(self):
        """Set up a small PagedKVCache and standard KV cache for testing."""
        self.batch_size = 1
        self.block_size = 64
        self.q_per_kv = 2 # Queries per key-value
        self.n_query_groups = 2 # Reduced num_heads in grouped-query attention
        self.head_dim = 128
        self.max_sequence_length = 256
        self.dtype = torch.float  # Ensure dtype consistency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize paged KV cache
        self.paged_cache = PagedKVCache(
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

    # def test_paged_sdpa_vs_torch_sdpa(self):
    #     """Test paged_sdpa output against PyTorch's built-in SDPA function."""
    #     tokens_to_store = 41  # Less than block size to test partial filling

    #     # Create random queries, keys, and values with matching dtype
    #     q = torch.randn((self.batch_size, self.q_per_kv, self.n_query_groups, tokens_to_store, self.head_dim),
    #                     dtype=self.dtype, device=self.device)
    #     k = torch.randn((self.batch_size, tokens_to_store, self.n_query_groups, self.head_dim),
    #                     dtype=self.dtype, device=self.device)
    #     v = torch.randn_like(k)

    #     # Add KV pairs to paged KV cache
    #     self.paged_cache.add_to_cache(k, v)

    #     # Store KV pairs in standard PyTorch KV cache
    #     self.kv_cache["keys"][:, :tokens_to_store, :, :] = k
    #     self.kv_cache["values"][:, :tokens_to_store, :, :] = v

    #     # Define attention mask (None = no mask)
    #     attn_mask = None

    #     # Run paged SDPA
    #     paged_sdpa_output = paged_sdpa(q, tokens_to_store, self.paged_cache, attn_mask, is_causal=False, dropout_p=0.0)

    #     # Ensure dtype consistency
    #     paged_sdpa_output = paged_sdpa_output.to(self.dtype)

    #     # Reshape q for PyTorch SDPA (needs heads in dim 2)
    #     q_reshaped = q.reshape(self.batch_size,  self.n_query_groups*self.q_per_kv,  tokens_to_store, self.head_dim)

    #     # Run PyTorch's built-in SDPA
    #     torch_sdpa_output = F.scaled_dot_product_attention(
    #         q_reshaped,
    #         self.kv_cache["keys"][:, :tokens_to_store, :, :].reshape(
    #             self.batch_size, self.n_query_groups, tokens_to_store, self.head_dim
    #         ),
    #         self.kv_cache["values"][:, :tokens_to_store, :, :].reshape(
    #             self.batch_size, self.n_query_groups, tokens_to_store, self.head_dim
    #         ),
    #         attn_mask=attn_mask,
    #         dropout_p=0.0,
    #         is_causal=False,
    #         enable_gqa = True
    #     ).reshape(self.batch_size, self.q_per_kv, self.n_query_groups, tokens_to_store, self.head_dim)

    #     # Ensure dtype consistency
    #     torch_sdpa_output = torch_sdpa_output.to(self.dtype)

    #     # Validate dtype consistency
    #     self.assertEqual(paged_sdpa_output.dtype, torch_sdpa_output.dtype, 
    #                      f"Mismatch in dtype: Paged SDPA ({paged_sdpa_output.dtype}) vs Torch SDPA ({torch_sdpa_output.dtype})")

    #     # Validate numerical correctness
    #     self.assertTrue(torch.allclose(paged_sdpa_output, torch_sdpa_output, atol=1e-3),
    #                     "Paged SDPA output does not match PyTorch SDPA output")

    def test_paged_sdpa_with_causal_mask(self):
        """Test paged_sdpa with a causal mask to ensure correct masking behavior."""
        tokens_to_store = 3

        # Create random queries, keys, and values
        q = torch.randn((self.batch_size, self.q_per_kv*self.n_query_groups, tokens_to_store, self.head_dim),
                        dtype=self.dtype, device=self.device).contiguous()
        k = torch.randn((self.batch_size, tokens_to_store, self.n_query_groups, self.head_dim),
                        dtype=self.dtype, device=self.device).contiguous()
        v = torch.randn_like(k).contiguous()

        # Add KV pairs to paged KV cache
        self.paged_cache.add_to_cache(k, v)
        
        print(self.paged_cache.block_tables[0].filled)
        
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

        print(f"paged_sdpa shae {paged_sdpa_output}")
        print(f"torch_sdpa shae {torch_sdpa_output}")
        # Validate dtype consistency
        self.assertEqual(paged_sdpa_output.dtype, torch_sdpa_output.dtype, 
                         f"Mismatch in dtype: Paged SDPA ({paged_sdpa_output.dtype}) vs Torch SDPA ({torch_sdpa_output.dtype})")

        # Validate numerical correctness
        self.assertTrue(torch.allclose(paged_sdpa_output, torch_sdpa_output, atol=1e-3),
                        "Paged SDPA output does not match PyTorch SDPA output with causal mask")

if __name__ == "__main__":
    unittest.main()
