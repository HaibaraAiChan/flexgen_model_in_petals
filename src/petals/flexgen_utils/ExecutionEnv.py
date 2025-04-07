import argparse
import dataclasses
from attr import define, field
from attr.setters import frozen
import functools
import gc
import math
import os
from typing import Tuple, Union, Optional, Any, Sequence, List
# fix recursive import
from petals.flexgen_utils.torch_device import TorchDevice, TorchDisk, TorchMixedDevice
from petals.flexgen_utils.base import fix_recursive_import

import numpy as np
import torch
@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    gpu: Any = None
    cpu: Any = None
    disk: Any = None
    mixed: Any = None

    @classmethod
    def create(cls, offload_dir):
        fix_recursive_import()  # Initialize TorchCompressedDevice
        gpu = TorchDevice("cuda:0")
        print('ExecutionEnv: gpu ', gpu)
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    def close_copy_threads(self):
        self.disk.close_copy_threads()
