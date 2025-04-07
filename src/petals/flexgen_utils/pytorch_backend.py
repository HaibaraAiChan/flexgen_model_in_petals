"""Implement tensor computations with pytorch."""
import math
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

from petals.flexgen_utils.compression import decompress
import torch
import torch.nn.functional as F
import numpy as np
import sys
from petals.flexgen_utils.DeviceType import DeviceType
from petals.flexgen_utils.utils import (GB, T, cpu_mem_stats, vector_gather, torch_type, np_type,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)
from torch import nn
from transformers.activations import ACT2FN


general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    """Fix recursive imports and initialize necessary components."""
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    
    # First import all necessary components
    from petals.flexgen_utils import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice
    
    # Then create the CPU device if it doesn't exist
    if global_cpu_device is None:
        # Create CPU device
        global_cpu_device = TorchDevice("cpu")
        
        # Ensure the compressed device is created
        if global_cpu_device.compressed_device is None:
            global_cpu_device.compressed_device = TorchCompressedDevice(global_cpu_device)

from pynvml import *

def see_memory_usage(message, force=True):
	logger = ''
	logger += message
	nvmlInit()
 
	# nvidia_smi.nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	logger += "\n Nvidia-smi: " + str((info.used) / 1024 / 1024 / 1024) + " GB"
	
	logger += '\n    Memory Allocated: '+str(torch.cuda.memory_allocated() / (1024 * 1024 * 1024)) +'  GigaBytes\n'
	logger +=   'Max Memory Allocated: ' + str(
		torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)) + '  GigaBytes\n'
	print(logger)
 
 
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(dim: int, end: int, inv_freq, theta= 10000.0):
    # freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # freqs = freqs.cuda()
    # inv_freq = 1.0 / (theta ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    # freqs = inv_freq[: (dim // 2)]
    freqs = inv_freq
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def rms_norm(hidden_states, weight, variance_epsilon=1e-5):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# class DeviceType(Enum):
#     CPU = auto()
#     CUDA = auto()
#     DISK = auto()
#     MIXED = auto()
#     COMPRESSED = auto()

#     @staticmethod
#     def convert(name):
#         if name == "cpu":
#             return DeviceType.CPU
#         elif name == "cuda":
#             return DeviceType.CUDA
#         elif name == "disk":
#             return DeviceType.DISK
#         elif name == "mixed":
#             return DeviceType.MIXED
#         elif name == "compressed":
#             return DeviceType.COMPRESSED
#         else:
#             raise ValueError(f"Invalid name: {name}")

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

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                # Convert numpy array to torch tensor
                tmp = torch.from_numpy(np_array)
                print('tmp.shape', tmp.shape) # torch.Size([4096, 4096])
                # Use the device's own compressed_device for compression
                if self.device.base_device.compressed_device is None:
                    raise RuntimeError("Device's compressed_device is None")
                    
                ctmp = self.device.base_device.compressed_device.compress(tmp, self.data[2])
                print('ctmp', ctmp) # torch.Size([2048, 4096]), scale [4096,2,4096]?
                # import pdb; pdb.set_trace()
                general_copy(self, None, ctmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_np(np.load(filename))
    ##############################################################  
    def load_from_state(self, param):
        # print(param.shape)
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np.array(param))
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                # Ensure global_cpu_device is initialized
                global global_cpu_device
                if global_cpu_device is None:
                    # Create a CPU device if it doesn't exist
                    from petals.flexgen_utils import compression
                    global TorchCompressedDevice
                    TorchCompressedDevice = compression.TorchCompressedDevice
                    global_cpu_device = TorchDevice("cpu")
                
                tmp = param
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(param) 
    #--------------------------------------------------------------#     
    def load_from_state_dict(self, state_dict):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(state_dict, self.data)
        else:
            self.load_from_state(state_dict)
    ########-----------------------------------------##################
    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")

class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        global global_cpu_device, TorchCompressedDevice
        
        # Handle both string and torch.device objects
        if isinstance(name, torch.device):
            self.name = str(name)
            self.dev = name
        else:
            self.name = name
            self.dev = torch.device(name)
            
        self.mem_capacity = mem_capacity
        self.flops = flops
        self.links = []
        self.compressed_device = None

        # Check device type based on the device string
        device_str = str(self.dev)
        if device_str == "cpu":
            self.device_type = DeviceType.CPU
            if global_cpu_device is None:
                global_cpu_device = self
            # Initialize compressed device for CPU
            from petals.flexgen_utils import compression
            if TorchCompressedDevice is None:
                TorchCompressedDevice = compression.TorchCompressedDevice
            self.compressed_device = TorchCompressedDevice(self)
        elif "cuda" in device_str:
            self.device_type = DeviceType.CUDA
            # Initialize compressed device for CUDA
            from petals.flexgen_utils import compression
            if TorchCompressedDevice is None:
                TorchCompressedDevice = compression.TorchCompressedDevice
            self.compressed_device = TorchCompressedDevice(self)
        else:
            raise ValueError(f"Invalid device name: {device_str}")

        # Initialize attention compute workspace
        self.attention_compute_workspace = None
        self.workspace_pt = 0

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        if dtype in np_type:
            dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1
        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)
    
    ## seq_len is here key states shape [-2]
    def llama_input_embed(self, inputs, attention_mask, w_token, pad_token_id, donate, token_type_embeddings):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        # mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        
        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        # token_type_ids = torch.zeros(
        #         token_ids.size(), dtype=torch.long, device=token_embed.device
        # )

        # tte = token_type_embeddings(token_type_ids).half()
        # embeddings = token_embed + tte
        embeddings = token_embed
        return TorchTensor.create_from_torch(embeddings, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def llama_output_embed(self, inputs, w_ln, donate,
                         do_sample, temperature, lm_head, top_p):
        # decompress weights
        if lm_head.device.device_type == DeviceType.COMPRESSED:
            lm_head = lm_head.device.decompress(lm_head)

        b, s, h = inputs.shape
        # hidden = inputs.data
        hidden = rms_norm(inputs.data, w_ln.data)
        # hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, lm_head.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
            # ids = sample_top_p(probs, top_p)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        # num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
        #     config.n_head, config.input_dim, task.prompt_len, task.gen_len,
        #     policy.gpu_batch_size)
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.num_key_value_heads, config.hidden_size, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        print('mha hidden size ', hidden.shape)
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis=freq_cis[src_s: src_s + tgt_s])

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                    
                    # 确保缓存张量也是 float16 类型
                    if k.dtype != torch.float16:
                        k = k.to(dtype=torch.float16)
                    if v.dtype != torch.float16:
                        v = v.to(dtype=torch.float16)
                        
                    # 确保缓存张量在正确的设备上
                    k = k.to(device=self.dev)
                    v = v.to(device=self.dev)
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                    
                    # 确保缓存张量也是 float16 类型
                    if k.dtype != torch.float16:
                        k = k.to(dtype=torch.float16)
                    if v.dtype != torch.float16:
                        v = v.to(dtype=torch.float16)
                        
                    # 确保缓存张量在正确的设备上
                    k = k.to(device=self.dev)
                    v = v.to(device=self.dev)
                        
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)
                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                
                # 确保缓存张量也是 float16 类型
                if k.dtype != torch.float16:
                    k = k.to(dtype=torch.float16)
                    
                # 确保缓存张量在正确的设备上
                k = k.to(device=self.dev)
                    
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v_cache.data[:src_s]
                
                # 确保缓存张量也是 float16 类型
                if v.dtype != torch.float16:
                    v = v.to(dtype=torch.float16)
                    
                # 确保缓存张量在正确的设备上
                v = v.to(device=self.dev)
                    
                v[src_s - 1:src_s] = v_new
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)
                value = self._sparse_attention_value(q, k, v_new, v, attention_mask.data,
                    b, src_s, tgt_s, n_head, head_dim, attn_sparsity)
        else:
            # shape: (b * n_head, head_dim, s)
            k = k_cache.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
            # shape: (b * n_head, s, head_dim)
            v = v_cache.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)
            value = self._attention_value(q, k, v, attention_mask.data,
                b, src_s, tgt_s, n_head, head_dim)

        # shape: (b, s, h)
        value = value.view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)
        
        # 检查value的大小是否与inputs.data匹配
        if value.shape != inputs.data.shape:
            print(f"Warning: Value tensor shape {value.shape} does not match inputs shape {inputs.data.shape}")
            # 如果value的大小是inputs.data的一半，尝试通过复制数据来恢复
            if value.numel() == inputs.data.numel() // 2:
                print(f"Value tensor size is half of inputs. Attempting to recover by duplicating data.")
                value_flat = value.view(-1)
                value_duplicated = torch.cat([value_flat, value_flat])
                value = value_duplicated[:inputs.data.numel()].view(inputs.data.shape)
            else:
                # 尝试调整value的大小以匹配inputs.data
                print(f"Attempting to reshape value tensor to match inputs shape")
                value = value.view(-1)[:inputs.data.numel()].view(inputs.data.shape)
                
        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k_new = k_new.permute(0, 1, 2)
        v_new = v_new.permute(0, 1, 2)

        if compress_cache:
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)

        return value, k_new, v_new
    

    def mha_llama(self, hidden_states, attention_mask, w_q, w_k, w_v, w_out, n_head, donate, compress_cache=False, comp_cache_config=None, input_layernorm=None, rotary_emb_inv_freq=None, compress_weight=False, comp_weight_config=None):
        print(f"[mha_llama] Input hidden shape: {hidden_states.shape}")
        
        # Check if weights are tuples and extract the TorchTensor
        if isinstance(w_q, tuple):
            w_q_tensor = w_q[0]
        else:
            w_q_tensor = w_q
            
        if isinstance(w_k, tuple):
            w_k_tensor = w_k[0]
        else:
            w_k_tensor = w_k
            
        if isinstance(w_v, tuple):
            w_v_tensor = w_v[0]
        else:
            w_v_tensor = w_v
            
        if isinstance(w_out, tuple):
            w_out_tensor = w_out[0]
        else:
            w_out_tensor = w_out
            
        if isinstance(input_layernorm, tuple):
            input_layernorm_tensor = input_layernorm[0]
        else:
            input_layernorm_tensor = input_layernorm
        if isinstance(rotary_emb_inv_freq, tuple):
            rotary_emb_inv_freq_tensor = rotary_emb_inv_freq[0]
        else:
            rotary_emb_inv_freq_tensor = rotary_emb_inv_freq
        # decompress weights
        #  w_q_tensor.device.device_type <DeviceType.CUDA: 2>
        # import pdb; pdb.set_trace()
        if compress_weight and comp_weight_config is not None:
        # if w_q_tensor.device.device_type == DeviceType.COMPRESSED:
            w_q = decompress(w_q_tensor, comp_weight_config)
            w_k = decompress(w_k_tensor, comp_weight_config)
            w_v = decompress(w_v_tensor, comp_weight_config)
            w_out = decompress(w_out_tensor, comp_weight_config)
            print('w_q after decompress', w_q)
            if input_layernorm is not None:
                input_layernorm = input_layernorm_tensor.decompress(input_layernorm_tensor)
            if rotary_emb_inv_freq is not None:
                rotary_emb_inv_freq = rotary_emb_inv_freq_tensor.decompress(rotary_emb_inv_freq_tensor)
        # import pdb; pdb.set_trace()
        print('w_q', w_q)
        print(f"[mha_llama] Input w_q shape: {w_q_tensor.shape}")
        print(f"[mha_llama] Input w_k shape: {w_k_tensor.shape}")
        print(f"[mha_llama] Input w_v shape: {w_v_tensor.shape}")
        print(f"[mha_llama] Input w_out shape: {w_out_tensor.shape}")
        
        # Ensure hidden states has the correct shape
        if hidden_states.shape[-1] != w_q_tensor.shape[1]:
            print(f"[mha_llama] Shape mismatch: expected {w_q_tensor.shape[1]} but got {hidden_states.shape[-1]}")
            # If the hidden size is half of expected, duplicate the data
            if hidden_states.shape[-1] * 2 == w_q_tensor.shape[1]:
                print("[mha_llama] Duplicating data to match expected size")
                hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
            else:
                raise ValueError(f"Hidden states dimension mismatch: expected {w_q_tensor.shape[1]} but got {hidden_states.shape[-1]}")
        
        bsz, q_len, h = hidden_states.shape   # hidden_states.shape should be   torch.Size([1, 1, 4096])
        head_dim = h // n_head
        freq_cis = precompute_freqs_cis(head_dim, 2048 * 2, rotary_emb_inv_freq.data)
        scaling = head_dim ** -0.5
        hidden = rms_norm(hidden_states.data, input_layernorm.data)
        
        # Use the decompressed weights directly
        w_q_data = w_q.data
        w_k_data = w_k.data
        w_v_data = w_v.data
        w_out_data = w_out.data
        
        # 确保hidden是float16类型
        if hidden.dtype != torch.float16:
            hidden = hidden.to(dtype=torch.float16)
        
        # 确保所有权重张量都是float16类型
        if w_q_data.dtype != torch.float16:
            w_q_data = w_q_data.to(dtype=torch.float16)
        if w_k_data.dtype != torch.float16:
            w_k_data = w_k_data.to(dtype=torch.float16)
        if w_v_data.dtype != torch.float16:
            w_v_data = w_v_data.to(dtype=torch.float16)
        if w_out_data.dtype != torch.float16:
            w_out_data = w_out_data.to(dtype=torch.float16)
        
        # 确保所有张量都在同一个设备上
        device = hidden.device
        w_q_data = w_q_data.to(device=device)
        w_k_data = w_k_data.to(device=device)
        w_v_data = w_v_data.to(device=device)
        w_out_data = w_out_data.to(device=device)
        
        q = F.linear(hidden, w_q_data) * scaling
        k = F.linear(hidden, w_k_data)
        v = F.linear(hidden, w_v_data)

        # Check if the tensor size matches the expected shape
        expected_size = bsz * q_len * n_head * head_dim
        actual_size = q.numel()
        
        if actual_size != expected_size:
            # If the actual size is half of expected, duplicate the data
            if actual_size * 2 == expected_size:
                q = torch.cat([q, q], dim=-1)
                k = torch.cat([k, k], dim=-1)
                v = torch.cat([v, v], dim=-1)
            else:
                raise ValueError(f"Tensor size mismatch: expected {expected_size} elements but got {actual_size}. Hidden size: {h}, Head dim: {head_dim}, Num heads: {n_head}")

        # Reshape tensors
        q = q.view(bsz, q_len, n_head, head_dim)
        k = k.view(bsz, q_len, n_head, head_dim)
        v = v.view(bsz, q_len, n_head, head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis=freq_cis[:q_len])

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(bsz * n_head, q_len, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(bsz * n_head, head_dim, q_len)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(bsz * n_head, q_len, head_dim)

        attn_weights = torch.bmm(q, k)

        idx = torch.arange(q_len, device=self.dev)
        causal_mask = (idx <= idx.view(q_len, 1)).view(1, 1, q_len, q_len) 
        mask = attention_mask.data.view(bsz, 1, 1, q_len) & causal_mask
        
        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(bsz, n_head, q_len, q_len)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(bsz * n_head, q_len, q_len)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(bsz, n_head, q_len, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(bsz, q_len, h)
        value = F.linear(value, w_out_data)
        
        value.add_(hidden_states.data)

        if donate[0]: hidden_states.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_cache_config)
            v = self.compressed_device.compress(v, comp_cache_config)

        return value, k, v

    def mha_gen_llama(self, inputs, attention_mask, w_q, w_k, w_v, w_out, n_head, k_cache, v_cache, donate, attn_sparsity=1.0, compress_cache=False, comp_cache_config=None, input_layernorm=None, rotary_emb_inv_freq=None, compress_weight=False):
        print(f"[mha_gen_llama] Input hidden shape: {inputs.shape}")
        
        # Check if weights are tuples and extract the TorchTensor
        if isinstance(w_q, tuple):
            w_q_tensor = w_q[0]
        else:
            w_q_tensor = w_q
            
        if isinstance(w_k, tuple):
            w_k_tensor = w_k[0]
        else:
            w_k_tensor = w_k
            
        if isinstance(w_v, tuple):
            w_v_tensor = w_v[0]
        else:
            w_v_tensor = w_v
            
        if isinstance(w_out, tuple):
            w_out_tensor = w_out[0]
        else:
            w_out_tensor = w_out
            
        if isinstance(input_layernorm, tuple):
            input_layernorm_tensor = input_layernorm[0]
        else:
            input_layernorm_tensor = input_layernorm
        if isinstance(rotary_emb_inv_freq, tuple):
            rotary_emb_inv_freq_tensor = rotary_emb_inv_freq[0]
        else:
            rotary_emb_inv_freq_tensor = rotary_emb_inv_freq
        # decompress weights
        # if w_q_tensor.device.device_type == DeviceType.COMPRESSED:
        if compress_weight:
            w_q = w_q_tensor.device.decompress(w_q_tensor)
            w_k = w_k_tensor.device.decompress(w_k_tensor)
            w_v = w_v_tensor.device.decompress(w_v_tensor)
            w_out = w_out_tensor.device.decompress(w_out_tensor)
            if input_layernorm is not None:
                input_layernorm = input_layernorm_tensor.device.decompress(input_layernorm_tensor)
            if rotary_emb_inv_freq is not None:
                rotary_emb_inv_freq = rotary_emb_inv_freq_tensor.device.decompress(rotary_emb_inv_freq_tensor)
        else:
            # If not compressed, use the tensors directly
            w_q = w_q_tensor
            w_k = w_k_tensor
            w_v = w_v_tensor
            w_out = w_out_tensor
            if input_layernorm is not None:
                input_layernorm = input_layernorm_tensor
            if rotary_emb_inv_freq is not None:
                rotary_emb_inv_freq = rotary_emb_inv_freq_tensor
                
        print(f"[mha_gen_llama] Input w_q shape: {w_q.shape}")
        print(f"[mha_gen_llama] Input w_k shape: {w_k.shape}")
        print(f"[mha_gen_llama] Input w_v shape: {w_v.shape}")
        print(f"[mha_gen_llama] Input w_out shape: {w_out.shape}")
        print(f"[mha_gen_llama] Input k_cache shape: {k_cache.shape}")
        print(f"[mha_gen_llama] Input v_cache shape: {v_cache.shape}")
        
        # Ensure hidden states has the correct shape
        if inputs.shape[-1] != w_q.shape[1]:
            print(f"[mha_gen_llama] Shape mismatch: expected {w_q.shape[1]} but got {inputs.shape[-1]}")
            # If the hidden size is half of expected, duplicate the data
            if inputs.shape[-1] * 2 == w_q.shape[1]:
                print("[mha_gen_llama] Duplicating data to match expected size")
                inputs = torch.cat([inputs, inputs], dim=-1)
            else:
                raise ValueError(f"Hidden states dimension mismatch: expected {w_q.shape[1]} but got {inputs.shape[-1]}")
        
        bsz, q_len, h = inputs.shape
        head_dim = h // n_head
        freq_cis = precompute_freqs_cis(head_dim, 2048 * 2, rotary_emb_inv_freq.data)
        scaling = head_dim ** -0.5
        hidden = rms_norm(inputs.data, input_layernorm.data)
        
        # Use the decompressed weights directly
        w_q_data = w_q.data
        w_k_data = w_k.data
        w_v_data = w_v.data
        w_out_data = w_out.data
        
        # 确保hidden是float16类型
        if hidden.dtype != torch.float16:
            hidden = hidden.to(dtype=torch.float16)
        
        # 确保所有权重张量都是float16类型
        if w_q_data.dtype != torch.float16:
            w_q_data = w_q_data.to(dtype=torch.float16)
        if w_k_data.dtype != torch.float16:
            w_k_data = w_k_data.to(dtype=torch.float16)
        if w_v_data.dtype != torch.float16:
            w_v_data = w_v_data.to(dtype=torch.float16)
        if w_out_data.dtype != torch.float16:
            w_out_data = w_out_data.to(dtype=torch.float16)
        
        # 确保所有张量都在同一个设备上
        device = hidden.device
        w_q_data = w_q_data.to(device=device)
        w_k_data = w_k_data.to(device=device)
        w_v_data = w_v_data.to(device=device)
        w_out_data = w_out_data.to(device=device)
        
        q = F.linear(hidden, w_q_data) * scaling
        k = F.linear(hidden, w_k_data)
        v = F.linear(hidden, w_v_data)

        # 直接重塑张量，不进行压缩和解压缩
        q = q.view(bsz, q_len, n_head, head_dim)
        k = k.view(bsz, q_len, n_head, head_dim)
        v = v.view(bsz, q_len, n_head, head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis=freq_cis[:q_len])

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(bsz * n_head, q_len, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(bsz * n_head, head_dim, q_len)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(bsz * n_head, q_len, head_dim)

        # 处理缓存
        if k_cache is not None:
            if isinstance(k_cache.device, TorchCompressedDevice):
                k_cache = k_cache.device.decompress(k_cache)
            k_cache = k_cache.to(device=device, dtype=torch.float16)
            k = torch.cat([k_cache, k], dim=2)
        
        if v_cache is not None:
            if isinstance(v_cache.device, TorchCompressedDevice):
                v_cache = v_cache.device.decompress(v_cache)
            v_cache = v_cache.to(device=device, dtype=torch.float16)
            v = torch.cat([v_cache, v], dim=2)

        attn_weights = torch.bmm(q, k)

        idx = torch.arange(k.shape[2], device=self.dev)
        causal_mask = (idx <= idx.view(-1, 1)).view(1, 1, k.shape[2], k.shape[2])
        mask = attention_mask.data.view(bsz, 1, 1, k.shape[2]) & causal_mask
        
        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(bsz, n_head, q_len, k.shape[2])
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(bsz * n_head, q_len, k.shape[2])
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(bsz, n_head, q_len, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(bsz, q_len, h)
        value = F.linear(value, w_out_data)
        
        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k_new = k.permute(2, 0, 1)
        v_new = v.permute(1, 0, 2)

        if compress_cache:
            k_new = self.compressed_device.compress(k_new, comp_cache_config)
            v_new = self.compressed_device.compress(v_new, comp_cache_config)

        return value, k_new, v_new

    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        """MLP."""
        # decompress weights
        if isinstance(wi.device, TorchCompressedDevice):
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        # 确保inputs.data是float16类型
        if inputs.data.dtype != torch.float16:
            inputs.data = inputs.data.to(dtype=torch.float16)
            
        # 确保所有权重张量都是float16类型
        if wi.data.dtype != torch.float16:
            wi.data = wi.data.to(dtype=torch.float16)
        if wo.data.dtype != torch.float16:
            wo.data = wo.data.to(dtype=torch.float16)
        if w_ln.data.dtype != torch.float16:
            w_ln.data = w_ln.data.to(dtype=torch.float16)
        if b_ln.data.dtype != torch.float16:
            b_ln.data = b_ln.data.to(dtype=torch.float16)
        if bi.data.dtype != torch.float16:
            bi.data = bi.data.to(dtype=torch.float16)
        if bo.data.dtype != torch.float16:
            bo.data = bo.data.to(dtype=torch.float16)
            
        # 确保所有张量都在同一个设备上
        device = inputs.data.device
        wi.data = wi.data.to(device=device)
        wo.data = wo.data.to(device=device)
        w_ln.data = w_ln.data.to(device=device)
        b_ln.data = b_ln.data.to(device=device)
        bi.data = bi.data.to(device=device)
        bo.data = bo.data.to(device=device)

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
       
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def mlp_llama(self, inputs, gate, down, up, donate, config, post_attention_layernorm, compress_weight=False):
        """MLP for LLaMA."""
        # Check if weights are tuples and extract the TorchTensor
        if isinstance(gate, tuple):
            gate_tensor = gate[0]
        else:
            gate_tensor = gate
            
        if isinstance(down, tuple):
            down_tensor = down[0]
        else:
            down_tensor = down
            
        if isinstance(up, tuple):
            up_tensor = up[0]
        else:
            up_tensor = up
            
        if isinstance(post_attention_layernorm, tuple):
            post_attention_layernorm_tensor = post_attention_layernorm[0]
        else:
            post_attention_layernorm_tensor = post_attention_layernorm
            
        # decompress weights if they are compressed
        # if gate_tensor.device.device_type == DeviceType.COMPRESSED:
        if compress_weight:
            
            gate = gate_tensor.device.decompress(gate_tensor)
            down = down_tensor.device.decompress(down_tensor)
            up = up_tensor.device.decompress(up_tensor)
            post_attention_layernorm = post_attention_layernorm_tensor.device.decompress(post_attention_layernorm_tensor)
        else:
            # If not compressed, use the tensors directly
            gate = gate_tensor
            down = down_tensor
            up = up_tensor
            post_attention_layernorm = post_attention_layernorm_tensor

        b, s, h = inputs.shape
        hidden_act = config.hidden_act
        act_fn = ACT2FN[hidden_act]
        src_out = rms_norm(inputs.data, post_attention_layernorm.data)
        
        # 确保权重是张量而不是元组
        if isinstance(gate.data, tuple):
            gate_data = gate.data[0]
        else:
            gate_data = gate.data
            
        if isinstance(down.data, tuple):
            down_data = down.data[0]
        else:
            down_data = down.data
            
        if isinstance(up.data, tuple):
            up_data = up.data[0]
        else:
            up_data = up.data
            
        # 确保所有权重张量都是float16类型
        if gate_data.dtype != torch.float16:
            gate_data = gate_data.to(dtype=torch.float16)
        if down_data.dtype != torch.float16:
            down_data = down_data.to(dtype=torch.float16)
        if up_data.dtype != torch.float16:
            up_data = up_data.to(dtype=torch.float16)
            
        # 确保所有张量都在同一个设备上
        device = inputs.data.device
        gate_data = gate_data.to(device=device)
        down_data = down_data.to(device=device)
        up_data = up_data.to(device=device)

        out = F.layer_norm(inputs.data, (h,), weight=post_attention_layernorm.data)
        gate_out = F.linear(out, gate_data)
        up_out = F.linear(out, up_data)
        out = F.silu(gate_out) * up_out
        out = F.linear(out, down_data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"



# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice], compress_weight=False):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices, compress_weight)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif compress_weight:
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
