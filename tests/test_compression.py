import unittest
import torch
import numpy as np
from petals.flexgen_utils.compression_utils import (
    get_compressed_tensor_indices,
    compress_tensor,
    decompress_tensor
)

class TestCompression(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test tensor
        self.test_tensor = torch.randn(100, 100)
        self.group_size = 64
        self.group_dim = 0
        self.num_bits = 4
        
    def test_standard_compression(self):
        print("\nTesting standard compression...")
        
        # Compress tensor
        compressed_data, scale = compress_tensor(
            self.test_tensor,
            self.num_bits,
            self.group_size,
            self.group_dim,
            compression_type="standard"
        )
        
        # Decompress tensor
        decompressed = decompress_tensor(
            compressed_data,
            scale,
            self.test_tensor.shape,
            self.group_size,
            self.group_dim,
            compression_type="standard"
        )
        
        # Check shapes
        self.assertEqual(decompressed.shape, self.test_tensor.shape)
        
        # Check that values are within expected range
        max_diff = torch.max(torch.abs(decompressed - self.test_tensor))
        self.assertLess(max_diff, 1.0)  # Allow for some quantization error
        
    def test_nf4_compression(self):
        print("\nTesting NF4 compression...")
        
        # Compress tensor
        compressed_data, scale = compress_tensor(
            self.test_tensor,
            self.num_bits,
            self.group_size,
            self.group_dim,
            compression_type="nf4"
        )
        
        # Decompress tensor
        decompressed = decompress_tensor(
            compressed_data,
            scale,
            self.test_tensor.shape,
            self.group_size,
            self.group_dim,
            compression_type="nf4"
        )
        
        # Check shapes
        self.assertEqual(decompressed.shape, self.test_tensor.shape)
        
        # Check that values are within expected range
        max_diff = torch.max(torch.abs(decompressed - self.test_tensor))
        self.assertLess(max_diff, 1.0)  # Allow for some quantization error
        
    def test_compression_indices(self):
        print("\nTesting compression indices...")
        
        # Test standard compression indices
        group_indices, element_indices = get_compressed_tensor_indices(
            self.test_tensor.shape,
            self.group_size,
            self.group_dim,
            compression_type="standard"
        )
        
        self.assertEqual(len(group_indices), (self.test_tensor.shape[0] + self.group_size - 1) // self.group_size)
        self.assertEqual(len(element_indices), self.test_tensor.numel())
        
        # Test NF4 compression indices
        group_indices, element_indices = get_compressed_tensor_indices(
            self.test_tensor.shape,
            self.group_size,
            self.group_dim,
            compression_type="nf4"
        )
        
        self.assertEqual(len(group_indices), (self.test_tensor.numel() + self.group_size - 1) // self.group_size)
        self.assertEqual(len(element_indices), self.test_tensor.numel())

if __name__ == '__main__':
    unittest.main() 