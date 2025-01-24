import torch
import unittest
from yalis import PagedKVCache

class TestPagedKVCache(unittest.TestCase):

    def setUp(self):
        """Setup a small PagedKVCache for testing."""
        self.batch_size = 8
        self.block_size = 64
        self.num_heads = 2
        self.head_dim = 128
        self.max_sequence_length = 256
        self.dtype = torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cache = PagedKVCache(
            batch_size=self.batch_size,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_sequence_length=self.max_sequence_length,
            dtype=self.dtype,
            device=self.device
        )

    def test_initialization(self):
        """Test that the cache initializes correctly."""
        self.assertEqual(self.cache.num_blocks, self.batch_size * (self.max_sequence_length // self.block_size))
        self.assertEqual(self.cache.keys.shape, (self.cache.num_blocks, self.block_size, self.num_heads, self.head_dim))
        self.assertEqual(self.cache.values.shape, self.cache.keys.shape)
        self.assertEqual(len(self.cache.block_tables), 0)


    def test_add_to_cache_single_batch(self):
        """Test adding a single batch of key-value pairs to the cache and verify stored values."""
        tokens_to_store = 41
        k = torch.randn((self.batch_size, tokens_to_store, self.num_heads, self.head_dim),
                        dtype=self.dtype, device=self.device)
        v = torch.randn_like(k)

        # Add to cache
        self.cache.add_to_cache(k, v)

        # Ensure block tables are initialized
        self.assertEqual(len(self.cache.block_tables), self.batch_size)

        for batch_id, bt in enumerate(self.cache.block_tables):
            self.assertGreater(len(bt.physical_block_values), 0,
                               f"Batch {batch_id} should have allocated at least one block.")
            self.assertEqual(sum(bt.filled), tokens_to_store,
                             f"Batch {batch_id} should have stored {tokens_to_store} tokens.")

            # Validate stored keys and values
            reconstructed_k = []
            reconstructed_v = []

            offset = 0  # Keeps track of the token position in k, v
            for block_id, physical_block_num in enumerate(bt.physical_block_values):
                num_tokens_in_block = bt.filled[block_id]
                stored_k = self.cache.keys[physical_block_num, 0:num_tokens_in_block, :, :]
                stored_v = self.cache.values[physical_block_num, 0:num_tokens_in_block, :, :]

                # Compare stored values with original values
                expected_k = k[batch_id, offset:offset + num_tokens_in_block, :, :]
                expected_v = v[batch_id, offset:offset + num_tokens_in_block, :, :]

                self.assertTrue(torch.allclose(stored_k, expected_k, atol=1e-5),
                                f"Mismatch in stored keys at batch {batch_id}, block {block_id}.")
                self.assertTrue(torch.allclose(stored_v, expected_v, atol=1e-5),
                                f"Mismatch in stored values at batch {batch_id}, block {block_id}.")

                reconstructed_k.append(stored_k)
                reconstructed_v.append(stored_v)
                offset += num_tokens_in_block

            # Ensure that reconstructed KV matches the original KV
            final_k = torch.cat(reconstructed_k, dim=0)  # Rebuild from blocks
            final_v = torch.cat(reconstructed_v, dim=0)

            self.assertTrue(torch.allclose(final_k, k[batch_id], atol=1e-5),
                            f"Final reconstructed keys do not match for batch {batch_id}.")
            self.assertTrue(torch.allclose(final_v, v[batch_id], atol=1e-5),
                            f"Final reconstructed values do not match for batch {batch_id}.")


    def test_partial_block_fill(self):
        """Test adding tokens that partially fill a block and then continue filling."""
        tokens_to_store = 50  # Less than block size
        k = torch.randn((self.batch_size, tokens_to_store, self.num_heads, self.head_dim),
                        dtype=self.dtype, device=self.device)
        v = torch.randn_like(k)

        # First add - partial fill
        self.cache.add_to_cache(k, v)

        # Verify partial fill correctness
        for batch_id, bt in enumerate(self.cache.block_tables):
            self.assertGreater(len(bt.physical_block_values), 0,
                               f"Batch {batch_id} should have at least one block after partial fill.")
            self.assertEqual(sum(bt.filled), tokens_to_store,
                             f"Batch {batch_id} partial fill mismatch (expected {tokens_to_store}).")

        # Add more tokens
        additional_tokens = 20
        k_new = torch.randn((self.batch_size, additional_tokens, self.num_heads, self.head_dim),
                            dtype=self.dtype, device=self.device)
        v_new = torch.randn_like(k_new)

        # Second add - continuing fill
        self.cache.add_to_cache(k_new, v_new)

        # Verify entire fill correctness
        total_tokens = tokens_to_store + additional_tokens
        for batch_id, bt in enumerate(self.cache.block_tables):
            self.assertEqual(sum(bt.filled), total_tokens,
                             f"Batch {batch_id} total tokens mismatch (expected {total_tokens}).")

            # Reconstruct the stored data for this batch
            reconstructed_k = []
            reconstructed_v = []

            offset = 0
            for block_id, physical_block_num in enumerate(bt.physical_block_values):
                num_tokens_in_block = bt.filled[block_id]
                stored_k = self.cache.keys[physical_block_num, 0:num_tokens_in_block, :, :]
                stored_v = self.cache.values[physical_block_num, 0:num_tokens_in_block, :, :]

                if offset < tokens_to_store:
                    # We are in the region of the first partial fill
                    expected_k = k[batch_id, offset : offset + num_tokens_in_block, :, :]
                    expected_v = v[batch_id, offset : offset + num_tokens_in_block, :, :]
                else:
                    # We are in the region of the second fill
                    offset_in_new = offset - tokens_to_store
                    expected_k = k_new[batch_id, offset_in_new : offset_in_new + num_tokens_in_block, :, :]
                    expected_v = v_new[batch_id, offset_in_new : offset_in_new + num_tokens_in_block, :, :]

                self.assertTrue(torch.allclose(stored_k, expected_k, atol=1e-5),
                                f"Mismatch in keys at batch {batch_id}, block {block_id}.")
                self.assertTrue(torch.allclose(stored_v, expected_v, atol=1e-5),
                                f"Mismatch in values at batch {batch_id}, block {block_id}.")

                reconstructed_k.append(stored_k)
                reconstructed_v.append(stored_v)
                offset += num_tokens_in_block

            # Final reconstruction
            final_k = torch.cat(reconstructed_k, dim=0)
            final_v = torch.cat(reconstructed_v, dim=0)

            # Build reference of what we expect (concatenation of old & new k)
            expected_k_full = torch.cat((k[batch_id], k_new[batch_id]), dim=0)
            expected_v_full = torch.cat((v[batch_id], v_new[batch_id]), dim=0)

            self.assertTrue(torch.allclose(final_k, expected_k_full, atol=1e-5),
                            f"Final reconstructed keys do not match for batch {batch_id} after partial fill.")
            self.assertTrue(torch.allclose(final_v, expected_v_full, atol=1e-5),
                            f"Final reconstructed values do not match for batch {batch_id} after partial fill.")


    def test_add_multiple_batches(self):
        """Test that multiple batches are stored separately, verifying stored data as well."""
        tokens_batch1 = 30
        tokens_batch2 = 45

        k1 = torch.randn((self.batch_size, tokens_batch1, self.num_heads, self.head_dim),
                         dtype=self.dtype, device=self.device)
        v1 = torch.randn_like(k1)

        k2 = torch.randn((self.batch_size, tokens_batch2, self.num_heads, self.head_dim),
                         dtype=self.dtype, device=self.device)
        v2 = torch.randn_like(k2)

        # First fill
        self.cache.add_to_cache(k1, v1)
        # Second fill
        self.cache.add_to_cache(k2, v2)

        # For each batch, total tokens should be sum
        total_tokens = tokens_batch1 + tokens_batch2

        for batch_id, bt in enumerate(self.cache.block_tables):
            self.assertEqual(sum(bt.filled), total_tokens,
                             f"Batch {batch_id} total tokens mismatch (expected {total_tokens}).")

            # Reconstruct keys and values
            reconstructed_k = []
            reconstructed_v = []

            offset = 0
            for block_id, physical_block_num in enumerate(bt.physical_block_values):
                num_tokens_in_block = bt.filled[block_id]
                stored_k = self.cache.keys[physical_block_num, :num_tokens_in_block, :, :]
                stored_v = self.cache.values[physical_block_num, :num_tokens_in_block, :, :]

                if offset < tokens_batch1:
                    # Belongs to first fill
                    expected_k = k1[batch_id, offset : min(tokens_batch1, offset + num_tokens_in_block), :, :]
                    expected_v = v1[batch_id, offset : min(tokens_batch1, offset + num_tokens_in_block), :, :]
                    # If partial block extends beyond batch1, also compare remainder with batch2
                    leftover = (offset + num_tokens_in_block) - tokens_batch1
                    if leftover > 0:
                        expected_k2 = k2[batch_id, 0 : leftover, :, :]
                        expected_v2 = v2[batch_id, 0 : leftover, :, :]
                        expected_k = torch.cat((expected_k, expected_k2), dim=0)
                        expected_v = torch.cat((expected_v, expected_v2), dim=0)
                else:
                    # Belongs to second fill
                    offset_in_batch2 = offset - tokens_batch1
                    expected_k = k2[batch_id, offset_in_batch2 : offset_in_batch2 + num_tokens_in_block, :, :]
                    expected_v = v2[batch_id, offset_in_batch2 : offset_in_batch2 + num_tokens_in_block, :, :]

                self.assertTrue(torch.allclose(stored_k, expected_k, atol=1e-5),
                                f"Mismatch in keys for batch {batch_id}, block {block_id}.")
                self.assertTrue(torch.allclose(stored_v, expected_v, atol=1e-5),
                                f"Mismatch in values for batch {batch_id}, block {block_id}.")

                reconstructed_k.append(stored_k)
                reconstructed_v.append(stored_v)
                offset += num_tokens_in_block

            # Verify final reconstruction
            final_k = torch.cat(reconstructed_k, dim=0)
            final_v = torch.cat(reconstructed_v, dim=0)

            # Build the ground truth
            expected_k_full = torch.cat((k1[batch_id], k2[batch_id]), dim=0)
            expected_v_full = torch.cat((v1[batch_id], v2[batch_id]), dim=0)

            self.assertTrue(torch.allclose(final_k, expected_k_full, atol=1e-5),
                            f"Mismatch in final reconstructed keys for batch {batch_id}.")
            self.assertTrue(torch.allclose(final_v, expected_v_full, atol=1e-5),
                            f"Mismatch in final reconstructed values for batch {batch_id}.")


    def test_eviction_logic(self):
        """Test that eviction happens when there are not enough free blocks."""
        small_cache = PagedKVCache(
            batch_size=self.batch_size,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_sequence_length=128,  # Only 2 blocks per batch
            dtype=self.dtype,
            device=self.device
        )

        tokens1 = 64
        tokens2 = 64  # Will need to evict the first one

        k1 = torch.randn((self.batch_size, tokens1, self.num_heads, self.head_dim),
                         dtype=self.dtype, device=self.device)
        v1 = torch.randn_like(k1)

        k2 = torch.randn((self.batch_size, tokens2, self.num_heads, self.head_dim),
                         dtype=self.dtype, device=self.device)
        v2 = torch.randn_like(k2)

        small_cache.add_to_cache(k1, v1)
        small_cache.add_to_cache(k2, v2)  # This should trigger eviction

        # Ensure that eviction happened
        self.assertLessEqual(len(small_cache.block_tables[0].physical_block_values), 2,
                             "Eviction logic failed: more blocks than expected after second fill.")

    def test_block_allocation_integrity(self):
        """Test that allocated blocks match expected counts."""
        tokens_to_store = 128  # Requires 2 full blocks
        k = torch.randn((self.batch_size, tokens_to_store, self.num_heads, self.head_dim),
                        dtype=self.dtype, device=self.device)
        v = torch.randn_like(k)

        self.cache.add_to_cache(k, v)

        for bt in self.cache.block_tables:
            self.assertEqual(len(bt.physical_block_values), 2,
                             "Expected exactly 2 blocks to be allocated.")
            self.assertEqual(sum(bt.filled), tokens_to_store,
                             "Number of tokens stored mismatch with tokens_to_store.")


    def test_allocate_blocks_function(self):
        """Test that the allocate_blocks function correctly assigns memory."""
        block_table = PagedKVCache.BlockTable(block_size=self.block_size)
        tokens_to_save = 96  # Requires more than one block

        allocated_blocks, partial_index = self.cache.allocate_blocks(tokens_to_save, block_table)

        expected_num_blocks = (tokens_to_save + self.block_size - 1) // self.block_size
        self.assertEqual(len(allocated_blocks), expected_num_blocks,
                         "allocate_blocks function allocated a different number of blocks than expected.")

    def test_get_max_k_cache_tokens(self):
        """Test retrieval of max K-cache tokens."""
        tokens_to_store = 75
        k = torch.randn((self.batch_size, tokens_to_store, self.num_heads, self.head_dim),
                        dtype=self.dtype, device=self.device)
        v = torch.randn_like(k)

        self.cache.add_to_cache(k, v)

        max_tokens = self.cache.get_max_k_cache_tokens()
        self.assertEqual(max_tokens, tokens_to_store,
                         "get_max_k_cache_tokens did not return the correct max token count.")

if __name__ == "__main__":
    unittest.main()
