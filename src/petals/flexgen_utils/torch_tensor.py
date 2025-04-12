"""
TorchTensor module for flexgen_utils.
"""

import torch
import numpy as np
import os
from itertools import count

from petals.flexgen_utils.shared_types import torch_dtype_to_np_dtype
from petals.flexgen_utils.DeviceType import DeviceType

class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        # if isinstance(data, torch.Tensor):
            # assert data.device == device.dev ####
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device
        self.name = name or f"tensor_{next(TorchTensor.name_count)}"

    @property
    def bytes(self):
        return np.prod(self.shape) * np.dtype(torch_dtype_to_np_dtype(self.dtype)).itemsize

    @classmethod
    def next_name(cls):
        return f"tensor_{next(TorchTensor.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name)

    def delete(self):
        if self.device.device_type == DeviceType.CUDA:
            del self.data
        elif self.device.device_type == DeviceType.DISK:
            if os.path.exists(self.data):
                os.remove(self.data)

    def load_from_np(self, np_array):
        """Load data from a numpy array."""
        if self.device.device_type == DeviceType.COMPRESSED:
            # For compressed tensors, we need to compress the numpy array first
            if hasattr(self.device, 'compress'):
                # Convert numpy array to torch tensor and ensure it's float16
                if np_array.dtype != np.float16:
                    print(f"Converting numpy array from {np_array.dtype} to float16")
                    np_array = np_array.astype(np.float16)
                
                # Convert to torch tensor and ensure it's float16
                torch_array = torch.from_numpy(np_array).to(self.device.base_device.dev)
                if torch_array.dtype != torch.float16:
                    print(f"Converting torch tensor from {torch_array.dtype} to float16")
                    torch_array = torch_array.to(torch.float16)
                
                # Get compression config
                comp_config = self.data[2]
                # Compress the tensor with the correct parameters
                compressed_data = self.device.compress(
                    torch_array, 
                    comp_config
                )
                # Copy the compressed data
                self.data[0].data.copy_(compressed_data.data[0].data)
                self.data[1].data.copy_(compressed_data.data[1].data)
            else:
                # If the device doesn't have a compress method, try to handle the size mismatch
                print(f"Warning: Loading numpy array with shape {np_array.shape} into compressed tensor with shape {self.shape}")
                # Check if the shapes are compatible after compression
                if np_array.shape[0] == self.shape[0] * 2:  # If the source is twice the size
                    # Use only the first half of the data
                    np_array = np_array[:self.shape[0]]
                    print(f"Using only the first half of the data: {np_array.shape}")
                # Try to load the data
                try:
                    self.data[0].data.copy_(torch.from_numpy(np_array).to(self.device.base_device.dev))
                except RuntimeError as e:
                    print(f"Error loading data: {e}")
                    print(f"Source shape: {np_array.shape}, Target shape: {self.data[0].data.shape}")
                    # Try to resize the data if possible
                    if len(np_array.shape) == len(self.data[0].data.shape):
                        # Resize the data to match the target shape
                        resized_data = np.resize(np_array, self.data[0].data.shape)
                        self.data[0].data.copy_(torch.from_numpy(resized_data).to(self.device.base_device.dev))
                    else:
                        raise
        else:
            # For non-compressed tensors, load directly
            self.data.copy_(torch.from_numpy(np_array).to(self.device.dev))

    def load_from_np_file(self, filename):
        np_array = np.load(filename)
        # Convert to float16 if needed
        if self.device.device_type == DeviceType.COMPRESSED and np_array.dtype != np.float16:
            print(f"Converting numpy array from {np_array.dtype} to float16 in load_from_np_file")
            np_array = np_array.astype(np.float16)
        self.load_from_np(np_array)

    def load_from_state(self, param):
        # print(param.shape)
        if self.device.device_type == DeviceType.CUDA:
            self.data.copy_(param.to(self.device.dev))
        elif self.device.device_type == DeviceType.DISK:
            np.save(self.data, param.detach().cpu().numpy())
        elif self.device.device_type == DeviceType.MIXED:
            # The tensor is on mixed devices, do recursive calls
            tensors, segment_points = self.data
            for i in range(len(tensors)):
                start, end = segment_points[i], segment_points[i+1]
                tensors[i].load_from_state(param[start:end])
        elif self.device.device_type == DeviceType.COMPRESSED:
            # The tensor is compressed, do recursive calls
            data, scale, compression_config = self.data
            data.load_from_state(param)

    def load_from_state_dict(self, state_dict):
        self.load_from_state(state_dict[self.name])

    def copy(self, dst, src_indices=None):
        """
        Copy from this tensor to the destination tensor or device.
        
        Args:
            dst: Destination tensor or device
            src_indices: Source indices to copy from
            
        Returns:
            The destination tensor
        """
        # 检查 dst 是否是 TorchDevice 对象
        if hasattr(dst, 'device_type') and not hasattr(dst, 'data'):
            # 如果 dst 是 TorchDevice，创建一个新的 TorchTensor
            from petals.flexgen_utils.torch_tensor import TorchTensor
            new_dst = TorchTensor(self.shape, self.dtype, None, dst, name=self.name)
            # 递归调用 smart_copy
            return self.smart_copy(new_dst, src_indices)
        
        return self.smart_copy(dst, src_indices)

    def smart_copy(self, dst, src_indices=None):
        """
        Smart copy from this tensor to the destination tensor.
        
        Args:
            dst: Destination tensor or device
            src_indices: Source indices to copy from
            
        Returns:
            The destination tensor
        """
        # 检查 dst 是否是 TorchDevice 对象
        if hasattr(dst, 'device_type') and not hasattr(dst, 'data'):
            # 如果 dst 是 TorchDevice，创建一个新的 TorchTensor
            from petals.flexgen_utils.torch_tensor import TorchTensor
            new_dst = TorchTensor(self.shape, self.dtype, None, dst, name=self.name)
            # 递归调用 smart_copy
            return self.smart_copy(new_dst, src_indices)
            
        # 检查源张量和目标张量的设备类型
        if self.device.device_type == DeviceType.COMPRESSED:
            # 源张量是压缩的
            if dst.device.device_type == DeviceType.COMPRESSED:
                # 目标张量也是压缩的，使用 general_copy_compressed
                from petals.flexgen_utils.compression import general_copy_compressed
                general_copy_compressed(dst, None, self, src_indices)
            else:
                # 目标张量不是压缩的，需要先解压
                # 获取压缩配置
                comp_config = self.data[2]
                
                # 创建压缩设备
                from petals.flexgen_utils.compression import TorchCompressedDevice
                compressed_device = TorchCompressedDevice(dst.device)
                
                # 解压源张量 - 传递压缩数据元组而不是整个self对象
                decompressed_tensor = compressed_device.decompress(self.data)
                
                # 确保 decompressed_tensor 和 decompressed_tensor.data 不为 None
                if decompressed_tensor is None:
                    raise ValueError("Failed to decompress tensor: decompressed_tensor is None")
                
                if decompressed_tensor.data is None:
                    raise ValueError("Failed to decompress tensor: decompressed_tensor.data is None")
                
                # 如果提供了源索引，应用它们
                if src_indices:
                    decompressed_tensor = decompressed_tensor[src_indices]
                
                # 确保 dst.data 不为 None
                if dst.data is None:
                    # 如果 dst.data 为 None，初始化它
                    dst.data = torch.zeros_like(decompressed_tensor.data, device=dst.device.dev)
                
                # 复制到目标张量
                dst.data.copy_(decompressed_tensor.data)
        elif hasattr(dst, 'device') and dst.device.device_type == DeviceType.COMPRESSED:
            # 目标张量是压缩的，但源张量不是
            # 检查 self.data 是否已经是压缩格式 (tensor, scale, compression_config)
            if isinstance(self.data, tuple) and len(self.data) == 3 and hasattr(self.data[2], 'compression_type'):
                # 如果 self.data 已经是压缩格式，直接使用它
                dst.data = self.data
            else:
                # 如果 self.data 不是压缩格式，需要先压缩
                # 如果 dst.data 为 None，需要先初始化它
                if dst.data is None:
                    from petals.flexgen_utils.compression import TorchCompressedDevice
                    compressed_device = TorchCompressedDevice(dst.device.base_device)
                    # 使用默认的压缩配置
                    from petals.flexgen_utils.compression import CompressionConfig
                    comp_config = CompressionConfig()
                else:
                    # 如果dst.data已经存在，使用其现有的压缩配置
                    comp_config = dst.data[2]
                    compressed_device = TorchCompressedDevice(dst.device.base_device)
                
                # 如果提供了源索引，应用它们
                if src_indices:
                    src_data = self.data[src_indices]
                else:
                    src_data = self.data
                
                # 压缩源张量并复制
                compressed_tensor = compressed_device.compress(src_data, comp_config)
                dst.data = compressed_tensor.data
        else:
            # 普通张量复制，dst is a TorchTensor
            # 检查 self.data 是否已经是压缩格式 (tensor, scale, compression_config)
            if isinstance(self.data, tuple) and len(self.data) == 3 and hasattr(self.data[2], 'compression_type'):
                # 如果 self.data 已经是压缩格式，需要先解压
                from petals.flexgen_utils.compression import TorchCompressedDevice
                compressed_device = TorchCompressedDevice(dst.device)
                
                # 解压源张量 - 传递压缩数据元组而不是整个self对象
                decompressed_tensor = compressed_device.decompress(self.data)
                
                # 确保 decompressed_tensor 和 decompressed_tensor.data 不为 None
                if decompressed_tensor is None:
                    raise ValueError("Failed to decompress tensor: decompressed_tensor is None")
                
                if decompressed_tensor.data is None:
                    raise ValueError("Failed to decompress tensor: decompressed_tensor.data is None")
                
                # 如果提供了源索引，应用它们
                if src_indices:
                    decompressed_tensor = decompressed_tensor[src_indices]
                
                # 确保 dst.data 不为 None
                if dst.data is None:
                    # 如果 dst.data 为 None，初始化它
                    dst.data = torch.zeros_like(decompressed_tensor.data, device=dst.device.dev)
                
                # 复制到目标张量
                dst.data.copy_(decompressed_tensor.data)
            else:
                # 普通张量复制
                # 确保 dst.data 不为 None
                if dst.data is None:
                    # 如果 dst.data 为 None，初始化它
                    if src_indices:
                        dst.data = torch.zeros_like(self.data[src_indices], device=dst.device.dev)
                    else:
                        dst.data = torch.zeros_like(self.data, device=dst.device.dev)
                
                # 复制数据
                if src_indices:
                    dst.data.copy_(self.data[src_indices])
                else:
                    dst.data.copy_(self.data)
                
        return dst

    def move(self, dst):
        """
        Move this tensor to the destination device.
        
        Args:
            dst: Destination device
            
        Returns:
            A new tensor on the destination device
        """
        # 创建一个新的张量，形状和数据类型与当前张量相同
        new_tensor = TorchTensor(self.shape, self.dtype, None, dst, name=self.name)
        
        # 使用 smart_copy 复制数据
        self.smart_copy(new_tensor)
        
        return new_tensor

    def __str__(self):
        return f"TorchTensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, device={self.device})" 