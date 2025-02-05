import unittest
import torch

# Adjust the import to match your code structure
from yalis import PagedKVCache

class TestPagedKVCache(unittest.TestCase):
    def setUp(self):
        """
        Create a small PagedKVCache with consistent dimensions.
        """
        self.batch_size = 2
        self.block_size = 4
        self.num_heads = 2
        self.head_dim = 3
        self.max_sequence_length = 100
        self.dtype = torch.float32
        self.device = torch.device("cuda")  # or "cuda"

        # For k_shape and v_shape, your code references only the last dimension
        # (k_shape[-1], v_shape[-1]) for the 'head_dim'. 
        # So we set them to something like (hidden_dim,) or (head_dim,).
        # If your code expects more complex shapes, adapt accordingly.
        self.k_shape = (self.head_dim,)  
        self.v_shape = (self.head_dim,)  # or possibly different if K/V dimension differ

        self.cache = PagedKVCache(
            k_shape=self.k_shape,
            v_shape=self.v_shape,
            batch_size=self.batch_size,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_sequence_length=self.max_sequence_length,
            dtype=self.dtype,
            device="cuda"
        )
    
    def reconstruct_keys_for_batch(self, batch_id, total_tokens):
        """
        Demonstration method: reconstruct the keys stored in the cache
        for a single batch (batch_id) up to 'total_tokens' tokens.

        This relies on:
         - self.cache.block_tables[batch_id].physical_block_values
         - self.cache.filled or block_table's "filled" data
         - self.cache.keys (the actual stored data)

        You may need to adapt this logic to match your metadata structure.
        """
        block_table = self.cache.block_tables[batch_id]
        block_indices = block_table.physical_block_values[:block_table.num_valid_blocks]  # e.g., a tensor or list
        # 'filled' might track how many tokens are used in each block index
        # or you might use self.cache.filled if that directly applies

        # We'll accumulate the tokens in a list, then cat at the end
        # We assume your block_table has a .filled array that parallels .physical_block_values
        # If not, adapt accordingly.
        all_tokens = []
        tokens_collected = 0

        for i, blk_id in enumerate(block_indices):
            # Get how many tokens in this block
            # e.g. if block_table has a .filled[i], or if you do self.cache.filled[blk_id]
            
            tokens_in_block = self.cache.filled[blk_id]

            # read that slice from self.cache.keys
            block_data = self.cache.keys[blk_id, :tokens_in_block, :, :]
            # print(f"{block_data}")
            all_tokens.append(block_data)
            tokens_collected += tokens_in_block

            if tokens_collected >= total_tokens:
                break

        # cat them along dimension=0 (sequence dimension) => shape: [total_tokens, num_heads, head_dim]
        if all_tokens:
            reconstructed = torch.cat(all_tokens, dim=0)
            # if we overshot, we can slice back to total_tokens
            reconstructed = reconstructed[:total_tokens]
        else:
            # no tokens?
            reconstructed = torch.empty((0, self.cache.num_heads, self.cache.head_dim))

        # final shape: [total_tokens, num_heads, head_dim]
        return reconstructed

    # def test_extended_memory_layout_correctness(self):
    #     """
    #     This extended test checks partial fill, full block usage,
    #     and final block usage for a multi-batch scenario, then
    #     reconstructs keys from the cache and compares with original.
    #     """
    #     # We'll pick a scenario that triggers partial, full, final:
    #     # - block_size=4
    #     # - batch_size=2, each with seq_len=6
    #     #    => each batch needs partial fill, then 1 full block, and leftover for final block
        
    #     seq_len = 6
    #     token_counter = torch.zeros(self.batch_size, dtype=torch.int, device=self.device)

    #     # Create random data
    #     k = torch.randn(self.batch_size, seq_len, self.num_heads, self.head_dim, device=self.device)
    #     v = torch.randn_like(k)

    #     # Add to cache
    #     self.cache.add_to_cache_new(k, v,initial_prompt_lengths = torch.tensor(prefill_seq_lens, device = self.device))

    #     # Now reconstruct keys for each batch, compare with original
    #     for b in range(self.batch_size):
    #         reconstructed_k = self.reconstruct_keys_for_batch(b, seq_len)
    #         original_k = k[b]  # shape: [seq_len, num_heads, head_dim]

    #         self.assertEqual(
    #             reconstructed_k.shape,
    #             original_k.shape,
    #             f"Reconstructed shape {reconstructed_k.shape} != original {original_k.shape} for batch {b}"
    #         )

    #         self.assertTrue(
    #             torch.allclose(reconstructed_k, original_k, atol=1e-5),
    #             f"Keys mismatch for batch {b}. They are not close enough."
    #         )

    def test_extended_multi_step_decode(self):
      """
      More thorough test:
        1. Prefill stage (padded batch, with one sequence having 3 tokens and another having 6).
        2. Multiple decode steps (1 token per batch element) using token_counter.
        3. Reconstruct from cache to confirm correctness.
      """
      batch_size = self.batch_size
      prefill_seq_lens = [3, 6]  # First batch element has 3 tokens, second has 6 tokens.
      max_prefill_seq_len = max(prefill_seq_lens)  # Padded length for prefill

      # -------------------
      # Prefill Stage Setup
      # -------------------
      k_prefill = torch.randn(batch_size, max_prefill_seq_len, self.num_heads, self.head_dim, device=self.device)
      v_prefill = torch.randn_like(k_prefill)

      # Prefill: Send the padded tensor
      self.cache.add_to_cache_new(k_prefill, v_prefill, initial_prompt_lengths = torch.tensor(prefill_seq_lens, device = self.device))

      # Token counter after prefill contains **actual sequence lengths**, not zero.
      token_counter = torch.tensor(prefill_seq_lens, dtype=torch.int, device=self.device)

      # ---------------------
      # Decode Stage (2 Steps)
      # ---------------------
      steps = 70

      # Store references to decode-only tokens so we can verify final results
      decode_tokens_k = []
      decode_tokens_v = []

      for step in range(steps):
          seq_len = 1  # Each decode step adds exactly 1 token per batch
          k_step = torch.randn(batch_size, seq_len, self.num_heads, self.head_dim, device=self.device)
          v_step = torch.randn_like(k_step)

          # Keep track of "supposed" decode tokens in the final sequence
          decode_tokens_k.append(k_step)
          decode_tokens_v.append(v_step)

          # Add decode tokens to cache (with token_counter)
          self.cache.add_to_cache_new(k_step, v_step, token_counter=token_counter)
          token_counter += seq_len  # Each batch element gets 1 more token
          print("__________________________________________________________________________________________________________________")

      # ---------------------
      # Final Reconstruction
      # ---------------------
      # The final total tokens per batch = prefill_seq_len + (steps * 1)
      total_tokens = [seq_len + steps for seq_len in prefill_seq_lens]

      # Merge the "expected" final keys:
      #  1) Prefill tokens (unpadded portion)
      #  2) Decoded tokens
      decode_k_all = torch.cat(decode_tokens_k, dim=1)  # shape [batch, steps, num_heads, head_dim]
      final_expected_k = []
      max_tokens = max(token_counter)
      for b in range(batch_size):
          original_seq_k = torch.cat((k_prefill[b, :prefill_seq_lens[b]], decode_k_all[b]), dim=0)
          
          final_expected_k.append(torch.cat((original_seq_k, torch.zeros(max_tokens-original_seq_k.size(0), original_seq_k.size(1), original_seq_k.size(2), device=self.device))))

      final_expected_k = torch.stack(final_expected_k)  # shape => [batch_size, total_tokens[b], num_heads, head_dim]
      print(f"{self.cache.filled}")
      # Reconstruct from cache and compare
      for b in range(batch_size):
          reconstructed_k = self.reconstruct_keys_for_batch(b, total_tokens[b])
          reconstructed_k = torch.cat((reconstructed_k, torch.zeros(max_tokens-reconstructed_k.size(0), reconstructed_k.size(1), reconstructed_k.size(2), device=self.device)))
        #   print(f"{reconstructed_k}")
          original_k = final_expected_k[b]  # shape [total_tokens[b], num_heads, head_dim]
        #   print(f"{original_k}")

          # Check shape
          self.assertEqual(
              reconstructed_k.shape,
              original_k.shape,
              f"Decode reconstruction shape mismatch for batch {b}"
          )

          # Check data correctness
          self.assertTrue(
              torch.allclose(reconstructed_k, original_k, atol=1e-5),
              f"Decode mismatch for batch {b}. The final reconstructed keys differ from original."
          )

    # def test_extended_multi_step_decode_into_next_block(self):
    #   """
    #   More thorough test:
    #     1. Prefill stage (padded batch, with one sequence having 3 tokens and another having 6).
    #     2. Multiple decode steps (1 token per batch element) using token_counter.
    #     3. Reconstruct from cache to confirm correctness.
    #   """
    #   batch_size = self.batch_size
    #   prefill_seq_lens = [3, 6, 3, 6, 3, 6, 3]  # First batch element has 3 tokens, second has 6 tokens.
    #   max_prefill_seq_len = max(prefill_seq_lens)  # Padded length for prefill

    #   # -------------------
    #   # Prefill Stage Setup
    #   # -------------------
    #   k_prefill = torch.randn(batch_size, max_prefill_seq_len, self.num_heads, self.head_dim, device=self.device)
    #   v_prefill = torch.randn_like(k_prefill)

    #   # Prefill: Send the padded tensor
    #   self.cache.add_to_cache_new(k_prefill, v_prefill)

    #   # Token counter after prefill contains **actual sequence lengths**, not zero.
    #   token_counter = torch.tensor(prefill_seq_lens, dtype=torch.int, device=self.device)

    #   # ---------------------
    #   # Decode Stage (2 Steps)
    #   # ---------------------
    #   steps = 7

    #   # Store references to decode-only tokens so we can verify final results
    #   decode_tokens_k = []
    #   decode_tokens_v = []

    #   for step in range(steps):
    #       seq_len = 1  # Each decode step adds exactly 1 token per batch
    #       k_step = torch.randn(batch_size, seq_len, self.num_heads, self.head_dim, device=self.device)
    #       v_step = torch.randn_like(k_step)

    #       # Keep track of "supposed" decode tokens in the final sequence
    #       decode_tokens_k.append(k_step)
    #       decode_tokens_v.append(v_step)

    #       # Add decode tokens to cache (with token_counter)
    #       self.cache.add_to_cache_new(k_step, v_step, token_counter=token_counter)
    #       token_counter += seq_len  # Each batch element gets 1 more token

    #   # ---------------------
    #   # Final Reconstruction
    #   # ---------------------
    #   # The final total tokens per batch = prefill_seq_len + (steps * 1)
    #   total_tokens = [seq_len + steps for seq_len in prefill_seq_lens]

    #   # Merge the "expected" final keys:
    #   #  1) Prefill tokens (unpadded portion)
    #   #  2) Decoded tokens
    #   decode_k_all = torch.cat(decode_tokens_k, dim=1)  # shape [batch, steps, num_heads, head_dim]
    #   final_expected_k = []
    #   max_tokens = max(token_counter)
    #   for b in range(batch_size):
    #       original_seq_k = torch.cat((k_prefill[b, :prefill_seq_lens[b]], decode_k_all[b]), dim=0)
          
    #       final_expected_k.append(torch.cat((original_seq_k, torch.zeros(max_tokens-original_seq_k.size(0), original_seq_k.size(1), original_seq_k.size(2), device=self.device))))

    #   final_expected_k = torch.stack(final_expected_k)  # shape => [batch_size, total_tokens[b], num_heads, head_dim]
    #   print(f"{self.cache.filled}")
    #   # Reconstruct from cache and compare
    #   for b in range(batch_size):
    #       reconstructed_k = self.reconstruct_keys_for_batch(b, total_tokens[b])
    #       reconstructed_k = torch.cat((reconstructed_k, torch.zeros(max_tokens-reconstructed_k.size(0), reconstructed_k.size(1), reconstructed_k.size(2), device=self.device)))
    #       print(f"{reconstructed_k}")
    #       original_k = final_expected_k[b]  # shape [total_tokens[b], num_heads, head_dim]
    #       print(f"{original_k}")

    #       # Check shape
    #       self.assertEqual(
    #           reconstructed_k.shape,
    #           original_k.shape,
    #           f"Decode reconstruction shape mismatch for batch {b}"
    #       )

    #       # Check data correctness
    #       self.assertTrue(
    #           torch.allclose(reconstructed_k, original_k, atol=1e-5),
    #           f"Decode mismatch for batch {b}. The final reconstructed keys differ from original."
    #       )


    # def test_lru_eviction_with_reconstruction(self):
    #     """
    #     Test that LRU eviction occurs and that the blocks that remain 
    #     still hold correct data for the newly added tokens.
    #     """
    #     # We have num_blocks=8, block_size=4 => can store up to 32 tokens fully
    #     batch_size = 1
    #     seq_len = 40  # triggers eviction beyond capacity
    #     token_counter = torch.zeros(batch_size, dtype=torch.int)

    #     # Original data
    #     k = torch.randn(batch_size, seq_len, self.num_heads, self.head_dim)
    #     v = torch.randn_like(k)

    #     self.cache.add_to_cache_new(k, v, token_counter=token_counter)

    #     used_blocks = (self.cache.filled > 0).sum().item()
    #     self.assertLessEqual(used_blocks, self.num_blocks, "Should not exceed total blocks after LRU.")

    #     # We only check that the last tokens are indeed in the cache. 
    #     # Suppose we want to check the final 32 tokens are definitely stored 
    #     # (the first 8 got evicted).
    #     last_32_tokens = seq_len - 32
    #     # Reconstruct from the cache for those final 32 tokens
    #     # (We assume the code only keeps the last 32 tokens if it's strictly LRU).
    #     needed_tokens = 32
    #     reconstructed_k = self.reconstruct_keys_for_batch(0, needed_tokens)
    #     original_k = k[0, last_32_tokens:, :, :]  # shape: [32, num_heads, head_dim]

    #     self.assertEqual(
    #         reconstructed_k.shape,
    #         original_k.shape,
    #         f"LRU shape mismatch: {reconstructed_k.shape} vs {original_k.shape}"
    #     )
    #     self.assertTrue(
    #         torch.allclose(reconstructed_k, original_k, atol=1e-5),
    #         "LRU mismatch: The final 32 tokens do not match what's in the cache."
    #     )

if __name__ == '__main__':
    unittest.main()
