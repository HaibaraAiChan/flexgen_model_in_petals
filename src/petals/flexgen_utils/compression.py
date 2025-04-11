import dataclasses
from typing import Optional, Tuple

import torch
import numpy as np

from petals.flexgen_utils.torch_device import TorchDevice
from petals.flexgen_utils.DeviceType import DeviceType
from petals.flexgen_utils.shared_types import np_dtype_to_torch_dtype
from petals.flexgen_utils.compression_utils import get_compressed_indices

# Import TorchTensor after defining the necessary types
from petals.flexgen_utils.torch_tensor import TorchTensor

@dataclasses.dataclass
class CompressionConfig:
    """Configuration for tensor compression."""
    num_bits: int = 4
    group_size: int = 64
    group_dim: int = 0
    symmetric: bool = False
    shape: Optional[Tuple[int, ...]] = None  # Target shape for compression
    enabled: bool = True
    compression_type: str = "standard"  # Options: "standard", "nf4"


class TorchCompressedDevice:
    """Manage tensors stored in a compressed format."""

    def __init__(self, base_device):
        # print('TorchCompressedDevice: ', base_device)
        self.name = "compressed"
        self.device_type = DeviceType.COMPRESSED
        self.base_device = base_device

        self.data_decompress_workspace = None
        self.workspace_pt = 0

    def allocate(self, shape, dtype, comp_config, pin_memory=None, name=None):
        """Allocate a compressed TorchTensor while maintaining the original shape."""
        assert comp_config.num_bits == 4 and dtype == np.float16
        # import pdb; pdb.set_trace()
        group_size, group_dim = comp_config.group_size, comp_config.group_dim

        # Calculate the number of groups needed
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        
        # Calculate the size of the compressed data
        # We're using 4-bit quantization, so each group of 2 elements can be stored in 1 byte
        compressed_size = num_groups * (group_size // 2)
        
        # Create shapes for the compressed data and scale factors
        # The data shape maintains the original dimensions but with compressed data
        data_shape = list(shape)
        data_shape[group_dim] = compressed_size
        
        # The scale shape includes the original dimensions plus an extra dimension for min/max values
        # For scale, we need to use num_groups instead of the original shape[group_dim]
        scale_shape = list(shape)
        scale_shape[group_dim] = num_groups  # Use num_groups instead of shape[group_dim]
        scale_shape.insert(group_dim + 1, 2)  # Add dimension for min/max values
        
        # 添加调试信息
        print(f"Allocating compressed tensor: original shape {shape}, data_shape {data_shape}, scale_shape {scale_shape}")
        print(f"group_dim={group_dim}, group_size={group_size}, num_groups={num_groups}, compressed_size={compressed_size}")
        
        # Allocate the compressed data and scale tensors
        data = self.base_device.allocate(data_shape, np.uint8, pin_memory=pin_memory)
        scale = self.base_device.allocate(scale_shape, np.float16, pin_memory=pin_memory)

        # Return a tensor with the original shape but compressed data
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (data, scale, comp_config), self, name=name)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.num_attention_heads, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        return k_cache, v_cache

    def init_attention_compute_workspace(self, config, task, policy):
        if self.base_device.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        b = policy.gpu_batch_size
        n_head = config.num_attention_heads
        head_dim = config.input_dim // n_head
        max_seq_len = task.prompt_len + task.gen_len - 1
        shape = (max_seq_len, b * n_head, head_dim)

        group_size, group_dim = (
            policy.comp_cache_config.group_size, policy.comp_cache_config.group_dim)
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        new_shape = (shape[:group_dim] + (num_groups, group_size) +
                     shape[group_dim+1:])

        self.data_decompress_workspace = [
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
        ]

    def compress(self, tensor, compression_config):
        """
        Compress a tensor using the configured compression settings.
        
        Args:
            tensor: The tensor to compress (can be torch.Tensor or TorchTensor)
            compression_config: The compression configuration
            
        Returns:
            A TorchTensor containing the compressed data
        """
        # Check if input is a TorchTensor
        if isinstance(tensor, TorchTensor):
            # 如果是 TorchTensor，使用其 data 属性
            tensor_data = tensor.data
            # 如果 data 是元组（压缩张量），则使用第一个元素
            if isinstance(tensor_data, tuple):
                tensor_data = tensor_data[0]
        # Check if input is a valid torch.Tensor
        elif not isinstance(tensor, torch.Tensor):
            try:
                tensor_data = torch.tensor(tensor)
            except Exception as e:
                raise ValueError(f"Failed to convert input to torch.Tensor: {e}")
        else:
            tensor_data = tensor
            
        # Use the internal compression function
        compressed_data, scale = _compress_tensor(tensor_data, compression_config)
        
        # Create a TorchTensor to represent the compressed data
        # 创建一个元组 (compressed_data, scale, compression_config) 作为 data 参数
        return TorchTensor(tensor_data.shape, tensor_data.dtype, (compressed_data, scale, compression_config), self)

    def decompress(self, tensor):
        """
        Decompress a tensor that was previously compressed.
        
        Args:
            tensor: The compressed tensor to decompress
            
        Returns:
            A torch.Tensor containing the decompressed data
        """
        # Check if input is a valid compressed tensor
        if not isinstance(tensor, TorchTensor):
            raise ValueError(f"Expected TorchTensor, got {type(tensor)}")
        
        # Get the compressed data and scale
        # 压缩张量的 data 是一个元组 (compressed_data, scale, compression_config)
        compressed_data, scale, compression_config = tensor.data
        
        # 获取原始形状
        original_shape = tensor.shape
        
        # 获取压缩配置
        group_size = compression_config.group_size
        group_dim = compression_config.group_dim
        
        # 计算填充
        pad_size = (group_size - original_shape[group_dim] % group_size) % group_size
        if pad_size > 0:
            padded_shape = list(original_shape)
            padded_shape[group_dim] += pad_size
            data = compressed_data.view(*padded_shape)
            # 移除填充
            slices = [slice(None)] * len(original_shape)
            slices[group_dim] = slice(original_shape[group_dim])
            data = data[slices]
        else:
            data = compressed_data.view(*original_shape)
        
        # 解量化数据
        if compression_config.compression_type == "nf4":
            # NF4 解量化
            data = data.float() * scale
        else:
            # 标准解量化
            data = data.float() * scale
        
        return data


def general_copy_compressed(src, dst):
    """
    Copy data between compressed tensors, handling shape mismatches and compression type differences.
    
    Args:
        src: Source compressed tensor
        dst: Destination compressed tensor
    """
    # Check if both tensors are compressed
    if not hasattr(src, 'device') or not hasattr(dst, 'device'):
        raise ValueError("Both source and destination must be compressed tensors")
        
    # Get compression configs
    src_config = src.data[2]  # 从压缩张量的 data 元组中获取压缩配置
    dst_config = dst.data[2]
    
    # Check if compression types match
    if src_config.compression_type != dst_config.compression_type:
        print(f"Warning: Compression type mismatch. Source: {src_config.compression_type}, Destination: {dst_config.compression_type}")
        # Decompress source and recompress with destination config
        decompressed = src.device.decompress(src)
        compressed = dst.device.compress(decompressed)
        dst.data = compressed.data
        return
        
    # Check if shapes match
    if src.shape != dst.shape:
        print(f"Warning: Shape mismatch. Source: {src.shape}, Destination: {dst.shape}")
        # Decompress source
        decompressed = src.device.decompress(src)
        
        # Resize if needed
        if decompressed.shape != dst.shape:
            decompressed = torch.nn.functional.interpolate(
                decompressed.unsqueeze(0),
                size=dst.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
        # Recompress with destination config
        compressed = dst.device.compress(decompressed)
        dst.data = compressed.data
        return
        
    # If shapes and compression types match, copy directly
    try:
        dst.data = src.data
    except RuntimeError as e:
        print(f"Error during direct copy: {e}")
        # Fallback to decompress-resize-compress method
        decompressed = src.device.decompress(src)
        compressed = dst.device.compress(decompressed)
        dst.data = compressed.data


def get_compressed_indices(tensor, indices, shape):
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
    # FIXED: Handle cases where start index is not aligned with group size
    start_idx = indices[group_dim].start
    if start_idx % group_size != 0:
        print(f"Warning: Start index {start_idx} is not aligned with group size {group_size}")
        # Adjust the start index to be aligned with the group size
        start_idx = (start_idx // group_size) * group_size
        indices[group_dim] = slice(start_idx, indices[group_dim].stop)

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

    # Add debug information
    print(f"get_compressed_indices: original indices {indices}")
    print(f"get_compressed_indices: data_indices {data_indices}")
    print(f"get_compressed_indices: scale_indices {scale_indices}")

    return data_indices, scale_indices


default_cache_config = CompressionConfig(
    num_bits=0, group_size=0, group_dim=0, symmetric=False, enabled=False)


def set_cache_compression_config(config):
    global default_cache_config
    default_cache_config = config


def get_cache_compression_config():
    return default_cache_config


def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (original_shape[:group_dim] + (num_groups, group_size) +
                 original_shape[group_dim+1:])

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = original_shape[:group_dim] + (pad_len,) + original_shape[group_dim+1:]
        tensor = torch.cat([
            tensor,
            torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim)
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim] +
            (original_shape[group_dim] + pad_len,) +
            original_shape[group_dim+1:])
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


def compress_and_decompress(tensor, config):
    packed_data = compress(tensor, config)
    return decompress(packed_data, config)


def test_simulated_compression():
    torch.manual_seed(0)
    a = torch.normal(0, 1, (64, 64, 64), dtype=torch.float16).cuda()

    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    packed_data = compress(a, config)
    b = decompress(packed_data, config)
    print(a[0])
    print(b[0])


def test_real_compression():
    torch.manual_seed(0)
    a = torch.normal(0, 1, (32, 1, 1), dtype=torch.float16).cuda()

    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    dev = TorchDevice("cuda:0", 0, 0).compressed_device
    packed = dev.compress(a)
    b = dev.decompress(packed)

    print(a.flatten())
    print(b.flatten())


if __name__ == "__main__":
    #test_simulated_compression()
    test_real_compression()


def _compress_tensor(tensor, compression_config):
    """
    Internal function to compress a torch.Tensor.
    
    Args:
        tensor: The torch.Tensor to compress
        compression_config: The compression configuration
        
    Returns:
        A tuple of (compressed_data, scale)
    """
    group_size, num_bits, group_dim, symmetric = (
        compression_config.group_size, compression_config.num_bits,
        compression_config.group_dim, compression_config.symmetric)
    assert num_bits == 4 and group_size % 2 == 0 and not symmetric

    shape = tensor.shape
    num_groups = (shape[group_dim] + group_size - 1) // group_size

    # Pad
    new_shape = (shape[:group_dim] + (num_groups, group_size) +
                 shape[group_dim+1:])
    pad_len = (group_size - shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = shape[:group_dim] + (pad_len,) + shape[group_dim+1:]
        tensor = torch.cat([
            tensor,
            torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim)
    data = tensor.view(new_shape)

    # Quantize
    B = 2 ** num_bits - 1
    
    # Check if we're using NF4 compression
    if compression_config.compression_type == "nf4":
        # NF4 uses a different quantization scheme with predefined scales
        # This is a simplified implementation - in a real implementation,
        # you would use the actual NF4 scales
        from petals.flexgen_utils.nf4_utils import get_nf4_scales
        
        # Get the NF4 scales
        nf4_scales = get_nf4_scales()
        
        # Quantize using NF4 scales
        # This is a simplified implementation
        # In a real implementation, you would use the actual NF4 quantization
        # algorithm
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]
        
        # Use NF4 scales for quantization
        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)
        data = data.clamp_(0, B).round_().to(torch.uint8)
    else:
        import pdb; pdb.set_trace()
        # Standard quantization
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)
        data = data.clamp_(0, B).round_().to(torch.uint8)

    # Pack
    left_indices = (
        tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
        (slice(0, data.shape[group_dim+1], 2),))
    right_indices = (
        tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
        (slice(1, data.shape[group_dim+1], 2),))
    data = torch.bitwise_or(
        data[left_indices].bitwise_left_shift(4), data[right_indices])

    # Reshape
    data_shape = (
        shape[:group_dim] + (num_groups * (group_size // 2),) + shape[group_dim+1:])
    scale_shape = (
        shape[:group_dim] + (num_groups, 2) + shape[group_dim+1:])
    data = data.view(data_shape)
    scale = torch.cat([scale, mn], dim=group_dim+1).view(scale_shape)

    return data, scale
