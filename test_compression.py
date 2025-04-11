import numpy as np
import torch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remove any existing FlexGen-Extension from sys.path
sys.path = [p for p in sys.path if 'FlexGen-Extension' not in p]

from petals.flexgen_utils.torch_device import TorchDevice
from petals.flexgen_utils.base import DeviceType, fix_recursive_import
from petals.flexgen_utils.torch_tensor import TorchTensor
from petals.flexgen_utils.compression import CompressionConfig

print("Python path:", sys.path)

# Fix recursive imports first
fix_recursive_import()

# Create a CPU device first to ensure global_cpu_device is initialized
cpu_device = TorchDevice('cpu')
print(f"CPU device type: {cpu_device.device_type}")
if cpu_device.compressed_device:
    print(f"CPU compressed device type: {cpu_device.compressed_device.device_type}")
else:
    print("Warning: CPU compressed device is None")

# Create a CUDA device
cuda_device = TorchDevice('cuda:0')
print(f"CUDA device type: {cuda_device.device_type}")
if cuda_device.compressed_device:
    print(f"CUDA compressed device type: {cuda_device.compressed_device.device_type}")
else:
    print("Warning: CUDA compressed device is None")

# Test standard compression
print("\nTesting standard compression:")
comp_config = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
shape = (10, 10)
dtype = np.float16
compressed_tensor = cuda_device.compressed_device.allocate(shape, dtype, comp_config)
print(f"Compressed tensor device type: {compressed_tensor.device.device_type}")

# Create a NumPy array
np_array = np.random.rand(*shape).astype(dtype)
compressed_tensor.load_from_np(np_array)
print("Successfully loaded NumPy array into compressed tensor (standard)")

# Test NF4 compression
print("\nTesting NF4 compression:")
nf4_config = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, compression_type="nf4")
nf4_compressed_tensor = cuda_device.compressed_device.allocate(shape, dtype, nf4_config)
print(f"NF4 compressed tensor device type: {nf4_compressed_tensor.device.device_type}")

# Load the same NumPy array into NF4 compressed tensor
nf4_compressed_tensor.load_from_np(np_array)
print("Successfully loaded NumPy array into compressed tensor (NF4)")

# Create a PyTorch tensor
torch_tensor = torch.randn(shape, dtype=torch.float16)
print("\nOriginal tensor:", torch_tensor)
print("Standard compressed tensor:", compressed_tensor)
print("NF4 compressed tensor:", nf4_compressed_tensor)

print("All tests passed!") 