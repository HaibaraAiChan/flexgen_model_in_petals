"""
TorchTensor module for flexgen_utils.
"""

import torch
import numpy as np
import os
from itertools import count
from typing import Optional, Union, Tuple

from petals.flexgen_utils.base import torch_dtype_to_np_dtype
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
        if self.device.device_type == DeviceType.CUDA:
            self.data.copy_(torch.from_numpy(np_array).to(self.device.dev))
        elif self.device.device_type == DeviceType.DISK:
            np.save(self.data, np_array)
        elif self.device.device_type == DeviceType.MIXED:
            # The tensor is on mixed devices, do recursive calls
            tensors, segment_points = self.data
            for i in range(len(tensors)):
                start, end = segment_points[i], segment_points[i+1]
                tensors[i].load_from_np(np_array[start:end])
        elif self.device.device_type == DeviceType.COMPRESSED:
            # The tensor is compressed, do recursive calls
            data, scale, compression_config = self.data
            data.load_from_np(np_array)

    def load_from_np_file(self, filename):
        np_array = np.load(filename)
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
        return self.smart_copy(dst, src_indices)

    def smart_copy(self, dst, src_indices=None):
        # This is a placeholder and will be implemented in a separate module
        pass

    def move(self, dst):
        # This is a placeholder and will be implemented in a separate module
        pass

    def __str__(self):
        return f"TorchTensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, device={self.device})" 