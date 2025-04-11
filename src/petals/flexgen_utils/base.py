"""
Base module for flexgen_utils containing shared types and utility functions.
"""

import torch
import numpy as np
from enum import Enum, auto
from itertools import count

from petals.flexgen_utils.torch_tensor import TorchTensor
from petals.flexgen_utils.DeviceType import DeviceType
from petals.flexgen_utils.shared_types import (
    torch_dtype_to_np_dtype, 
    np_dtype_to_torch_dtype, 
    general_copy_compressed,
    global_disk_device,
    SEG_DIM
)

from typing import Tuple

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

def general_copy(dst, dst_indices, src, src_indices):
    """
    Copy data between two tensors.
    
    Args:
        dst: Destination tensor
        dst_indices: Destination indices
        src: Source tensor
        src_indices: Source indices
    """
    # 检查源张量和目标张量是否都是压缩张量
    if (hasattr(src, 'device') and src.device.device_type == DeviceType.COMPRESSED and
            hasattr(dst, 'device') and dst.device.device_type == DeviceType.COMPRESSED):
        from petals.flexgen_utils.compression import general_copy_compressed
        general_copy_compressed(dst, dst_indices, src, src_indices)
        return

    # 检查源张量是否是压缩张量，而目标张量不是
    if (hasattr(src, 'device') and src.device.device_type == DeviceType.COMPRESSED and
            (not hasattr(dst, 'device') or dst.device.device_type != DeviceType.COMPRESSED)):
        # 解压源张量
        src_decompressed = src.device.decompress(src)
        # 使用解压后的张量进行复制
        src = src_decompressed

    # 检查目标张量是否是压缩张量，而源张量不是
    if ((not hasattr(src, 'device') or src.device.device_type != DeviceType.COMPRESSED) and
            hasattr(dst, 'device') and dst.device.device_type == DeviceType.COMPRESSED):
        # 压缩源张量
        if hasattr(dst.device, 'compress'):
            # 如果目标设备有compress方法，直接使用
            dst.device.compress(src, dst.device.compression_config)
        else:
            # 否则，创建一个压缩设备
            from petals.flexgen_utils.compression import TorchCompressedDevice
            compressed_device = TorchCompressedDevice(dst.device.base_device)
            compressed_data = compressed_device.compress(src, compressed_device.compression_config)
            # 复制压缩数据到目标张量
            dst.data[0].data.copy_(compressed_data.data[0].data)
            dst.data[1].data.copy_(compressed_data.data[1].data)
        return

    # 检查源张量和目标张量的形状是否匹配
    # 确保 src 和 dst 都是 TorchTensor 对象，而不是 TorchCompressedDevice 对象
    if not hasattr(src, 'shape') or not hasattr(dst, 'shape'):
        print(f"Warning: general_copy called with non-TorchTensor objects: src={type(src)}, dst={type(dst)}")
        # 如果 dst 是 TorchCompressedDevice，尝试使用其 compress 方法
        if hasattr(dst, 'compress'):
            # 获取压缩配置
            compression_config = getattr(dst, 'compression_config', None)
            if compression_config is None:
                # 如果没有压缩配置，尝试从设备获取
                if hasattr(dst, 'device') and hasattr(dst.device, 'compression_config'):
                    compression_config = dst.device.compression_config
                else:
                    # 如果仍然没有，使用默认配置
                    from petals.flexgen_utils.compression import CompressionConfig
                    compression_config = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, compression_type="nf4")
            
            # 使用获取到的压缩配置
            dst.compress(src, compression_config)
            return
        # 如果 src 是 TorchCompressedDevice，尝试使用其 decompress 方法
        if hasattr(src, 'decompress'):
            src_decompressed = src.decompress()
            dst.data.copy_(src_decompressed.data)
            return
        # 如果都不是，尝试直接复制
        try:
            dst.data.copy_(src.data)
            return
        except Exception as e:
            print(f"Error in fallback copy: {e}")
            raise

    src_shape = src.shape
    dst_shape = dst.shape
    
    if src_shape != dst_shape:
        print(f"Warning: Shape mismatch in general_copy: src_shape={src_shape}, dst_shape={dst_shape}")
        
        # 调整源张量的大小以匹配目标张量
        import torch.nn.functional as F
        src_resized = F.interpolate(
            src.unsqueeze(0).unsqueeze(0), 
            size=(dst_shape[0], src_shape[1]),
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).squeeze(0)
        
        # 使用调整后的张量进行复制
        src = src_resized

    # 执行复制操作
    try:
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)
    except RuntimeError as e:
        print(f"Error in general_copy: {e}")
        print(f"src.shape: {src.shape}, dst.shape: {dst.shape}")
        print(f"src_indices: {src_indices}, dst_indices: {dst_indices}")
        raise

def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])
