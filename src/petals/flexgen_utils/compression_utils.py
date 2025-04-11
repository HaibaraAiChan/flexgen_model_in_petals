"""
Compression utility functions for flexgen_utils.
"""

from typing import Tuple, List, Any, Optional, Union
from petals.flexgen_utils.DeviceType import DeviceType
import numpy as np
import torch

def get_compressed_indices(tensor, indices, shape):
    """
    Get indices for compressed tensor data and scale.
    
    Args:
        tensor: The compressed tensor
        indices: The indices to apply
        shape: The shape of the tensor
        
    Returns:
        Tuple of data indices and scale indices
    """
    comp_config = tensor.data[2]
    group_size, group_dim = comp_config.group_size, comp_config.group_dim
    assert comp_config.num_bits == 4

    if indices is None:
        indices = list(slice(0, x) for x in shape[:group_dim+1])
    else:
        indices = list(indices) + [slice(0, x) for x in shape[len(indices):]]
    
    # Calculate the compressed size for the specified dimension
    num_groups = (shape[group_dim] + group_size - 1) // group_size
    compressed_size = num_groups * (group_size // 2)
    
    # Ensure the start index is aligned with the group size
    assert indices[group_dim].start % group_size == 0

    # Create indices for the compressed data
    data_indices = list(indices)
    # Adjust the end index to account for compression
    data_indices[group_dim] = slice(
        indices[group_dim].start // 2, 
        min((indices[group_dim].stop + 1) // 2, compressed_size))

    # Create indices for the scale factors
    scale_indices = list(indices)
    scale_indices.insert(group_dim+1, slice(0, 2))
    scale_indices[group_dim] = slice(
        indices[group_dim].start // group_size,
        min((indices[group_dim].stop + group_size - 1) // group_size, num_groups))

    return data_indices, scale_indices 

def get_compressed_tensor_indices(
    shape: Tuple[int, ...],
    group_size: int,
    group_dim: int,
    compression_type: str = "standard"
) -> Tuple[np.ndarray, np.ndarray]:
    """Get indices for compressed tensor storage.
    
    Args:
        shape: Shape of the original tensor
        group_size: Size of compression groups
        group_dim: Dimension along which to group
        compression_type: Type of compression ("standard" or "nf4")
        
    Returns:
        Tuple of (group_indices, element_indices)
    """
    if compression_type == "nf4":
        # NF4 uses a different grouping strategy
        total_elements = np.prod(shape)
        num_groups = (total_elements + group_size - 1) // group_size
        
        group_indices = np.arange(num_groups)
        element_indices = np.arange(total_elements)
        
        return group_indices, element_indices
    else:
        # Standard compression grouping
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        
        group_indices = np.arange(num_groups)
        element_indices = np.arange(np.prod(shape))
        
        return group_indices, element_indices

def compress_tensor(
    tensor: Union[torch.Tensor, np.ndarray],
    num_bits: int,
    group_size: int,
    group_dim: int,
    compression_type: str = "standard"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress a tensor using the specified compression method.
    
    Args:
        tensor: Input tensor to compress
        num_bits: Number of bits for compression
        group_size: Size of compression groups
        group_dim: Dimension along which to group
        compression_type: Type of compression ("standard" or "nf4")
        
    Returns:
        Tuple of (compressed_data, scale)
    """
    if compression_type == "nf4":
        # NF4 compression implementation
        tensor = tensor.reshape(-1)  # Flatten
        num_groups = (tensor.numel() + group_size - 1) // group_size
        
        # Pad if necessary
        if tensor.numel() % group_size != 0:
            pad_size = group_size - (tensor.numel() % group_size)
            tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        
        # Reshape into groups
        tensor = tensor.reshape(num_groups, group_size)
        
        # Calculate scales for each group
        max_abs = torch.max(torch.abs(tensor), dim=1, keepdim=True)[0]
        scale = max_abs / ((1 << (num_bits - 1)) - 1)
        
        # Quantize
        tensor = torch.clamp(tensor / scale, -((1 << (num_bits - 1)) - 1), (1 << (num_bits - 1)) - 1)
        tensor = tensor.to(torch.int8)
        
        return tensor, scale
    else:
        # Standard compression implementation
        original_shape = tensor.shape
        tensor = tensor.reshape(-1)  # Flatten
        
        # Group elements
        num_groups = (tensor.numel() + group_size - 1) // group_size
        if tensor.numel() % group_size != 0:
            pad_size = group_size - (tensor.numel() % group_size)
            tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        
        tensor = tensor.reshape(num_groups, group_size)
        
        # Calculate scales
        max_abs = torch.max(torch.abs(tensor), dim=1, keepdim=True)[0]
        scale = max_abs / ((1 << (num_bits - 1)) - 1)
        
        # Quantize
        tensor = torch.clamp(tensor / scale, -((1 << (num_bits - 1)) - 1), (1 << (num_bits - 1)) - 1)
        tensor = tensor.to(torch.int8)
        
        return tensor, scale

def decompress_tensor(
    compressed_data: torch.Tensor,
    scale: torch.Tensor,
    original_shape: Tuple[int, ...],
    group_size: int,
    group_dim: int,
    compression_type: str = "standard"
) -> torch.Tensor:
    """Decompress a tensor using the specified compression method.
    
    Args:
        compressed_data: Compressed tensor data
        scale: Scale factors for each group
        original_shape: Shape of the original tensor
        group_size: Size of compression groups
        group_dim: Dimension along which to group
        compression_type: Type of compression ("standard" or "nf4")
        
    Returns:
        Decompressed tensor
    """
    if compression_type == "nf4":
        # NF4 decompression
        tensor = compressed_data.float() * scale
        tensor = tensor.reshape(-1)
        
        # Remove padding if necessary
        original_size = np.prod(original_shape)
        if tensor.numel() > original_size:
            tensor = tensor[:original_size]
        
        return tensor.reshape(original_shape)
    else:
        # Standard decompression
        tensor = compressed_data.float() * scale
        tensor = tensor.reshape(-1)
        
        # Remove padding if necessary
        original_size = np.prod(original_shape)
        if tensor.numel() > original_size:
            tensor = tensor[:original_size]
        
        return tensor.reshape(original_shape) 