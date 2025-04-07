"""
Base module for flexgen_utils containing shared types and utility functions.
"""

import torch
import numpy as np
from enum import Enum, auto
from itertools import count



# Shared constants
GB = 1024**3
T = 1024**4

# Utility functions
def torch_dtype_to_np_dtype(dtype):
    """Convert PyTorch dtype to NumPy dtype."""
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.float16:
        return np.float16
    elif dtype == torch.int32:
        return np.int32
    elif dtype == torch.int64:
        return np.int64
    elif dtype == torch.bool:
        return np.bool_
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def np_dtype_to_torch_dtype(dtype):
    """Convert NumPy dtype to PyTorch dtype."""
    if dtype == np.float32:
        return torch.float32
    elif dtype == np.float16:
        return torch.float16
    elif dtype == np.int32:
        return torch.int32
    elif dtype == np.int64:
        return torch.int64
    elif dtype == np.bool_:
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

# Fix recursive import function
def fix_recursive_import():
    """Fix recursive imports by ensuring modules are imported in the correct order."""
    # This function is a placeholder and will be implemented in a separate module
    pass 