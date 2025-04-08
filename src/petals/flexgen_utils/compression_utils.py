"""
Compression utility functions for flexgen_utils.
"""

from typing import Tuple, List, Any
from petals.flexgen_utils.DeviceType import DeviceType

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