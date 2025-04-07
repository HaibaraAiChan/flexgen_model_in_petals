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

# Create a compression config
comp_config = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)

# Create a compressed tensor on CUDA
shape = (10, 10)
dtype = np.float16
compressed_tensor = cuda_device.compressed_device.allocate(shape, dtype, comp_config)
print(f"Compressed tensor device type: {compressed_tensor.device.device_type}")

# Create a NumPy array
np_array = np.random.rand(*shape).astype(dtype)

# Load the NumPy array into the compressed tensor
# This should now work without errors
compressed_tensor.load_from_np(np_array)
print("Successfully loaded NumPy array into compressed tensor")

# Create a PyTorch tensor
torch_tensor = torch.randn(shape, dtype=torch.float16)

# Load the PyTorch tensor into the compressed tensor
# This should now work without errors
compressed_tensor.load_from_state(torch_tensor)
print("Successfully loaded PyTorch tensor into compressed tensor")

print("All tests passed!") 