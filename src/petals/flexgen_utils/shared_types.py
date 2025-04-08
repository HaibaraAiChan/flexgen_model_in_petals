"""
Shared types and constants for flexgen_utils.
"""

import torch
import numpy as np
from enum import Enum, auto
from itertools import count
from typing import Dict, Any, Optional, List, Tuple, Protocol

# Shared constants
SEG_DIM = 1  # Segment dimension for tensor operations
GB = 1024**3
T = 1024**4

# Type conversion mappings
torch_dtype_to_np_dtype: Dict[torch.dtype, np.dtype] = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

np_dtype_to_torch_dtype: Dict[np.dtype, torch.dtype] = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,
    np.uint8: torch.uint8,
    np.int32: torch.int32,
    np.int64: torch.int64,
}

# Protocol classes for type hints
class TorchDeviceProtocol(Protocol):
    name: str
    device_type: 'DeviceType'
    mem_capacity: Optional[int]
    flops: Optional[float]
    dev: torch.device
    compressed_device: Optional['TorchCompressedDeviceProtocol']
    mixed_device: Optional['TorchMixedDeviceProtocol']
    disk_device: Optional['TorchDiskProtocol']

class TorchCompressedDeviceProtocol(Protocol):
    name: str
    device_type: 'DeviceType'
    base_device: TorchDeviceProtocol

class TorchMixedDeviceProtocol(Protocol):
    name: str
    device_type: 'DeviceType'
    base_devices: List[TorchDeviceProtocol]

class TorchDiskProtocol(Protocol):
    name: str
    device_type: 'DeviceType'
    path: str
    mem_capacity: Optional[int]

# Global shared variables
general_copy_compressed = None
TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None
global_cuda_devices = {} 