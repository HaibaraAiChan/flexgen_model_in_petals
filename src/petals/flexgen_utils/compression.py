import dataclasses

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
    """Group-wise quantization."""
    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True


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
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
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
        n_head = config.n_head
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

    def compress(self, tensor, comp_config):
        """Compress a torch.Tensor while maintaining the original shape."""
        # Ensure we have a valid base device
        if self.base_device is None:
            if global_cpu_device is None:
                # Create CPU device first
                global_cpu_device = TorchDevice("cpu")
                # global_cpu_device will be set in TorchDevice.__init__
            self.base_device = global_cpu_device

        group_size, num_bits, group_dim, symmetric = (
            comp_config.group_size, comp_config.num_bits,
            comp_config.group_dim, comp_config.symmetric)
        assert num_bits == 4 and group_size % 2 == 0 and not symmetric

        if tensor.device.type == "cpu" and tensor.dtype == torch.float16:
            tensor = tensor.float()

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

        # Calculate the size of the compressed data
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
        print(f"Compressing tensor: original shape {shape}, data_shape {data_shape}, scale_shape {scale_shape}")
        print(f"group_dim={group_dim}, group_size={group_size}, num_groups={num_groups}, compressed_size={compressed_size}")
        
        # Reshape the data and scale tensors
        data = data.view(data_shape)
        scale = torch.cat([scale, mn], dim=group_dim+1).view(scale_shape)

        data = TorchTensor.create_from_torch(data, self.base_device)
        scale = TorchTensor.create_from_torch(scale, self.base_device)

        return TorchTensor(shape, tensor.dtype,
                           (data, scale, comp_config), self)

    def decompress(self, tensor):
        data, scale, comp_config = tensor.data
        group_size, num_bits, group_dim, symmetric = (
            comp_config.group_size, comp_config.num_bits,
            comp_config.group_dim, comp_config.symmetric)

        # Calculate the number of groups from the compressed data shape
        compressed_size = data.shape[group_dim]
        group_size_c = group_size // 2
        num_groups = compressed_size // group_size_c

        # Create a shape for the unpacked data
        shape = data.shape
        new_shape = list(shape)
        new_shape[group_dim] = num_groups * group_size
        
        # Pad if necessary
        pad_len = (group_size - tensor.shape[group_dim] % group_size) % group_size
        if pad_len != 0:
            pad_shape = shape[:group_dim] + (pad_len,) + shape[group_dim+1:]
            data = torch.cat([
                data,
                torch.zeros(pad_shape, dtype=data.dtype, device=data.device)],
                dim=group_dim)
        
        # Reshape for unpacking
        unpack_shape = list(shape)
        unpack_shape[group_dim] = num_groups * group_size_c
        packed = data.data.view(unpack_shape)

        # Unpack
        if self.base_device.device_type == DeviceType.CPU:
            self.workspace_pt = (self.workspace_pt + 1) % len(
                self.data_decompress_workspace)
            data = self.data_decompress_workspace[
                self.workspace_pt][:shape[0]]
        else:
            new_shape = (shape[:group_dim] + (num_groups, group_size,) +
                         shape[group_dim+1:])
            data = torch.empty(new_shape, dtype=torch.float16, device=packed.device)
        
        # Unpack the 4-bit values
        left_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(0, data.shape[group_dim+1], 2),))
        right_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(1, data.shape[group_dim+1], 2),))
        data[left_indices] = packed.bitwise_right_shift(4)
        data[right_indices] = packed.bitwise_and(0xF)

        # Dequantize
        scale, mn = scale.data.split(1, dim=group_dim + 1)
        data.div_(scale)
        data.add_(mn)

        # Reshape back to the original shape
        unpad_len = (group_size - tensor.shape[group_dim] % group_size) % group_size
        if unpad_len != 0:
            flatten_shape = (shape[:group_dim] + (num_groups * group_size,) +
                             shape[group_dim+1:])
            indices = [slice(0, x) for x in flatten_shape]
            indices[group_dim] = slice(0, flatten_shape[group_dim] - unpad_len)
            data = data.view(flatten_shape)[indices].contiguous()

        return data.view(tensor.shape)


def general_copy_compressed(dst, dst_indices, src, src_indices):
    assert (src.device.device_type == DeviceType.COMPRESSED and
            dst.device.device_type == DeviceType.COMPRESSED)

    src_data_indices, src_scale_indices = get_compressed_indices(
        src, src_indices, src.shape)

    dst_data_indices, dst_scale_indices = get_compressed_indices(
        dst, dst_indices, dst.shape)
    
    # 添加调试信息
    print(f"Copying compressed tensor: src shape {src.shape}, dst shape {dst.shape}")
    print(f"src_data shape: {src.data[0].shape}, dst_data shape: {dst.data[0].shape}")
    print(f"src_data_indices: {src_data_indices}, dst_data_indices: {dst_data_indices}")
    
    # 检查源张量和目标张量的形状是否匹配
    src_data_shape = src.data[0].shape
    dst_data_shape = dst.data[0].shape
    
    # 检查索引是否会导致形状不匹配
    src_slice = src_data_indices[0]  # 假设 group_dim=0
    dst_slice = dst_data_indices[0]
    
    src_size = src_slice.stop - src_slice.start
    dst_size = dst_slice.stop - dst_slice.start
    
    if src_size != dst_size:
        print(f"Warning: Shape mismatch in general_copy_compressed: src_size={src_size}, dst_size={dst_size}")
        
        # 调整源索引以匹配目标索引的大小
        if src_size > dst_size:
            # 如果源大小大于目标大小，截断源
            src_data_indices[0] = slice(src_slice.start, src_slice.start + dst_size)
            src_scale_indices[0] = slice(src_scale_indices[0].start, src_scale_indices[0].start + (dst_size + 63) // 64)
        else:
            # 如果源大小小于目标大小，扩展目标
            dst_data_indices[0] = slice(dst_slice.start, dst_slice.start + src_size)
            dst_scale_indices[0] = slice(dst_scale_indices[0].start, dst_scale_indices[0].start + (src_size + 63) // 64)
    
    # 执行复制操作
    try:
        from petals.flexgen_utils.base import general_copy
        general_copy(dst.data[0], dst_data_indices, src.data[0], src_data_indices)
        general_copy(dst.data[1], dst_scale_indices, src.data[1], src_scale_indices)
    except RuntimeError as e:
        print(f"Error in general_copy_compressed: {e}")
        print(f"src.data[0].shape: {src.data[0].shape}, dst.data[0].shape: {dst.data[0].shape}")
        print(f"src_data_indices: {src_data_indices}, dst_data_indices: {dst_data_indices}")
        raise


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
    packed = dev.compress(a, config)
    b = dev.decompress(packed)

    print(a.flatten())
    print(b.flatten())


if __name__ == "__main__":
    #test_simulated_compression()
    test_real_compression()
