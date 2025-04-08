import argparse
import dataclasses


from typing import  Any
# fix recursive import
from petals.flexgen_utils.torch_device import TorchDevice 
from petals.flexgen_utils.torch_disk import TorchDisk
from petals.flexgen_utils.torch_mixed_device import TorchMixedDevice

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
        gpu = TorchDevice("cuda:0")
        print('ExecutionEnv: gpu ', gpu)
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    def close_copy_threads(self):
        self.disk.close_copy_threads()
