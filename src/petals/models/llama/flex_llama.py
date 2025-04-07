"""
Usage:
python3 -m flexgen.flex_llama --model huggingface repo --gpu-batch-size 32 --percent 100 0 100 0 100 0
modified on flex_opt.py
"""

import argparse
import dataclasses
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from petals.flexgen_utils.compression import CompressionConfig
from petals.flexgen_utils.llama_config import LlamaConfig, get_llama_config, download_llama_weights
from petals.flexgen_utils.base import fix_recursive_import, DeviceType
from petals.flexgen_utils.torch_device import TorchDevice, TorchDisk, TorchMixedDevice
from petals.flexgen_utils.torch_tensor import TorchTensor
from petals.flexgen_utils.task import Task
from petals.flexgen_utils.ExecutionEnv import ExecutionEnv
from petals.flexgen_utils.utils import ( GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_dtype_to_np_dtype, np_dtype_to_torch_dtype)
from torch import nn
from transformers import AutoTokenizer
from petals.flexgen_utils.timer import timers
from transformers.models.llama.modeling_llama import LlamaRMSNorm

fix_recursive_import()

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    # LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    repeat_kv,
    rotate_half,
)




DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent

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





def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]

def init_weight_list(weight_specs, policy, env):
    
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    print('dev_percents :[ disk, cpu, gpu]', dev_percents)
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    petals_weights=[]
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        # print('mid_percent ', mid_percent)
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]
        # print('weight_specs[i] ', weight_specs[i][2])
        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)
            # print('weight.shape', weight.shape)
            # print('weight', weight)

            if DUMMY_WEIGHT not in filename:
                try:
                    weight.load_from_np_file(weight_specs[i][2])
                except (FileNotFoundError, AttributeError) as e:
                    print(f"Warning: Could not load weight from file {weight_specs[i][2]}: {e}")
                    # 如果文件不存在或加载失败，使用随机初始化
                    weight.load_from_np(np.random.rand(*shape).astype(dtype))
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            # 检查压缩设备是否可用
            if hasattr(home, 'compressed_device') and home.compressed_device is not None:
                # 使用原始形状，不进行调整
                weight = home.compressed_device.allocate(
                    shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

                if DUMMY_WEIGHT not in filename:
                    try:
                        weight.load_from_np_file(weight_specs[i][2])
                    except (FileNotFoundError, AttributeError) as e:
                        print(f"Warning: Could not load weight from file {weight_specs[i][2]}: {e}")
                        # 如果文件不存在或加载失败，使用随机初始化
                        for i in range(2):
                            x = weight.data[i]
                            x.load_from_np(np.random.rand(*x.shape).astype(torch_dtype_to_np_dtype[x.dtype]))
                else:
                    for i in range(2):
                        x = weight.data[i]
                        x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))
            else:
                # 如果压缩设备不可用，回退到非压缩方式
                print(f"Warning: Compressed device not available, falling back to non-compressed allocation")
                weight = home.allocate(shape, dtype, pin_memory=pin_memory)
                if DUMMY_WEIGHT not in filename:
                    try:
                        weight.load_from_np_file(weight_specs[i][2])
                    except (FileNotFoundError, AttributeError) as e:
                        print(f"Warning: Could not load weight from file {weight_specs[i][2]}: {e}")
                        # 如果文件不存在或加载失败，使用随机初始化
                        weight.load_from_np(np.random.rand(*shape).astype(dtype))
                else:
                    weight.load_from_np(np.ones(shape, dtype))
        print('weight.data ', weight.data) # tuple: (data, scale, comp_config)
        print('weight ', weight)
        ret.append(weight)
        petals_weights.append(weight.data)
        
    return ret, petals_weights

# 添加一个新函数，用于从 PyTorch 模型加载权重到 FlexGen 格式
def load_weights_from_pytorch_model(model, policy, env, weight_home, block_index):
    """
    从 PyTorch 模型加载权重到 FlexGen 格式
    
    Args:
        model: PyTorch 模型
        policy: FlexGen 策略
        env: FlexGen 环境
        weight_home: 权重存储位置
        block_index: 块索引
    """
    weight_specs = []
    
    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        # 创建权重规格
        shape = param.shape
        dtype = param.dtype
        # 使用参数名称作为文件名，确保唯一性
        filename = f"block_{block_index}_{name}"
        
        weight_specs.append((shape, dtype, filename))
        
        # 将参数移动到 CPU，避免在 GPU 上存储
        param.data = param.data.to('cpu')
    
    try:
        # 初始化权重列表
        weights = init_weight_list(weight_specs, policy, env)
        # print('weights ', weights)
        petals_weights = []
        # 将权重加载到模型中
        for (name, _), weight in zip(model.named_parameters(), weights):
            param = getattr(model, name)
            param.data = weight.data.to(param.device)
            print('param', param)
            petals_weights.append(param.data)
        # 存储权重规格，以便后续使用
        weight_home[block_index] = weight_specs
        print('petals_weights ', petals_weights)
        return petals_weights
    except Exception as e:
        print(f"Warning: Failed to initialize weights with FlexGen: {e}")
        print("Falling back to direct parameter assignment")
        
        # 如果 FlexGen 初始化失败，直接使用参数赋值
        for name, param in model.named_parameters():
            # 确保参数在 CPU 上
            param.data = param.data.to('cpu')
        
        # 存储空的权重规格
        weight_home[block_index] = []
        
        return []




class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        
        self.self_attn = FLEX_LlamaAttention(config=config, env=env, policy=policy, layer_id=layer_id)
        self.mlp = FLEX_LlamaMLP(
            layer_id=layer_id,
            env=env,
            policy=policy,
            config=config
        )
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.compute = self.self_attn.compute
        self.policy = policy

    def set_task(self, task):
        self.self_attn.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        petals_attention_weights = self.self_attn.init_weight(home1, path)
        petals_mlp_weights = self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))
        return petals_attention_weights, petals_mlp_weights

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.self_attn.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
           weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.self_attn.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i):
        self.self_attn.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.self_attn.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val
        # Self Attention
        self.self_attn.forward(
            hidden=hidden,
            attention_mask=attention_mask,
            cache_read_buf=cache_read_buf,
            cache_write_buf=cache_write_buf,
            weight_read_buf=read_buf1,
            i=i,
            k=k
        )
        self.mlp.forward(hidden, cache_read_buf=None, cache_write_buf=None, weight_read_buf=read_buf2, i=i, k=k, attention_mask=attention_mask)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        if hasattr(self.hidden[i][j][k].val, 'data'):
            self.hidden[i][j][k].val = self.hidden[i][j][k].val.data
        
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                               self.weight_read_buf[j], self.attention_mask[k],
                               self.cache_write_buf[j][k], i, k)

class LlamaLM:
    def __init__(self, config, env: ExecutionEnv, path: str, policy: Policy):
        if isinstance(config, str):
            config = get_llama_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(InputEmbed(config=self.config, env=self.env, policy=self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(LlamaAttention(config=self.config, env=self.env, policy=self.policy, layer_id=i))
                layers.append(LlamaMLP(config=self.config, env=self.env, policy=self.policy, layer_id=i))
            else:
                layers.append(LlamaDecoderLayer(config=self.config, env=self.env, policy=self.policy, layer_id=i))
        layers.append(OutputEmbed(config=self.config, env=self.env, policy=self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None

        # Initialize weights and apply final processing
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_llama_weights(self.config.name, self.path)
        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return
        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos - 1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j - 1][k].pop()
            if hasattr(val, 'data'):
                val = val.data
            val = val.move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            hidden_val = self.hidden[i][j][k].pop()
            if hasattr(hidden_val, 'data'):
                hidden_val = hidden_val.data
            ids = hidden_val.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos + 1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos + 1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                if hasattr(x.val, 'data'):
                    x.val = x.val.data
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        if hasattr(self.hidden[i][j][k].val, 'data'):
            self.hidden[i][j][k].val = self.hidden[i][j][k].val.data
        
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                               self.weight_read_buf[j], self.attention_mask[k],
                               self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(
        self,
        inputs,
        max_new_tokens: int=32,
        do_sample: bool=True,
        temperature: float=0.6,
        stop: Optional[int] = None,
        debug_mode: Optional[str] = None,
        cut_gen_len: Optional[int] = None,
        top_p: float = 0.9,
        verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            top_p=top_p
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename

def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = [
        # "Simply put, the theory of relativity states that ",

        "I believe the meaning of life is",

        # """Translate English to French:
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
    ]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts

def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = '[PAD]'
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len
    # Task and policy
    # warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    ## weight and cache compression
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=1, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"
    llama_config = get_llama_config(args.model)
    cache_size = llama_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = llama_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {llama_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = LlamaLM(llama_config, env, args.path, policy)
    try:
        # print("warmup - generate")
        # output_ids = model.generate(
        #     warmup_inputs,
        #     max_new_tokens=1,
        #     debug_mode=args.debug_mode,
        #     verbose=args.verbose)
        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs,
            max_new_tokens=gen_len,
            debug_mode=args.debug_mode,
            cut_gen_len=cut_gen_len, 
            verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        llama_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="huggyllama/llama-7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="/tmp/data/llama_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="/tmp/data/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[50, 50, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)

class FLEX_LlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        super().__init__(config)
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight else self.compute)
        self.task = None
        self.temp_hidden_states = ValueHolder()

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, np.float16)
        i, dtype = (self.config.intermediate_size, np.float16)
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # 4 weight files
            # gate_proj
            ((i, h), dtype, path + "mlp.gate_proj.weight"),
            # down_proj
            ((h, i), dtype, path + "mlp.down_proj.weight"),
            # up_proj
            ((i, h), dtype, path + "mlp.up_proj.weight"),
            # post attention layer norm
            ((h, ), dtype, path + "post_attention_layernorm.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        gate, down, up, post_attention_layernorm = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                gate.smart_copy(dst1),
                down.smart_copy(dst1),
                up.smart_copy(dst1),
                post_attention_layernorm.smart_copy(dst2)
            ))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def forward(
        self,
        hidden_states,
        cache_read_buf,
        weight_read_buf,
        attention_mask,
        cache_write_buf,
        position_ids,
        k: int = 0
        ):
        donate = [False] * 9
        h, donate[0] = hidden_states.val.data if hasattr(hidden_states.val, 'data') else hidden_states.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((gate, donate[1]), (down, donate[3]),
             (up, donate[5]), (post_attention_layernorm, donate[7])) = weight_read_buf.pop()
        else:
            ((gate, _), (down, _),
             (up, _), (post_attention_layernorm, _)) = weight_read_buf.val

        # Access .data attribute if available for each weight tensor
        gate = gate.data if hasattr(gate, 'data') else gate
        down = down.data if hasattr(down, 'data') else down
        up = up.data if hasattr(up, 'data') else up
        post_attention_layernorm = post_attention_layernorm.data if hasattr(post_attention_layernorm, 'data') else post_attention_layernorm

        # If the weights are compressed, decompress them
        if hasattr(self.compute, 'compressed_device') and self.compute.compressed_device is not None:
            gate = self.compute.compressed_device.decompress(gate)
            down = self.compute.compressed_device.decompress(down)
            up = self.compute.compressed_device.decompress(up)
            post_attention_layernorm = self.compute.compressed_device.decompress(post_attention_layernorm)

        h = self.compute.mlp_llama(h, gate, down, up, donate, self.config, post_attention_layernorm)
        hidden_states.val = h
        self.temp_hidden_states.val = h
        
        # Return a tuple containing only the hidden states tensor
        return h

class FLEX_LlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        super().__init__(config)
        self.config = config
        self.llama_config = get_llama_config('huggyllama/llama-7b')
        self.num_heads = config.num_attention_heads
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute else self.env.gpu)
        self.task = None
        self.temp_hidden_states = ValueHolder()

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, np.float16)
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # 5 weight files
            # w_q
            ((h, h), dtype, path + "self_attn.q_proj.weight"),
            # w_k
            ((h, h), dtype, path + "self_attn.k_proj.weight"),
            # w_v
            ((h, h), dtype, path + "self_attn.v_proj.weight"),
            # w_out
            ((h, h), dtype, path + "self_attn.o_proj.weight"),
            # input layer norm
            ((h, ), dtype, path + "input_layernorm.weight"),
            # rotary_embed
            ((64, ), dtype, path + "self_attn.rotary_emb.inv_freq"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        # In LlamaDecoderLayer, weight_home.val contains (home1, home2)
        # We need to access the weights stored in home1
        if isinstance(weight_home.val, tuple) and len(weight_home.val) == 2:
            # This is the case when called from LlamaDecoderLayer
            home1, _ = weight_home.val
            # Check if home1 is a ValueHolder or a list
            if hasattr(home1, 'val'):
                w_q, w_k, w_v, w_out, input_layernorm, rotary_embed = home1.val
            else:
                # If home1 is a list, use it directly
                w_q, w_k, w_v, w_out, input_layernorm, rotary_embed = home1
        else:
            # This is the case when called directly
            w_q, w_k, w_v, w_out, input_layernorm, rotary_embed = weight_home.val
            
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_out.smart_copy(dst1),
                input_layernorm.smart_copy(dst2),
                rotary_embed.smart_copy(dst2)
            ))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.task is None:
            return
        
        shape = (self.task.prompt_len + self.task.gen_len,
                self.policy.gpu_batch_size * self.llama_config.n_head,
                self.config.hidden_size // self.llama_config.n_head)
        k_cache = self.env.gpu.allocate(shape, np.float16)
        v_cache = self.env.gpu.allocate(shape, np.float16)
        cache_home.store((k_cache, v_cache))

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                        k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices,  compress_weight=self.policy.compress_weight )

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices,compress_weight=self.policy.compress_weight)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices, self.policy.compress_weight)
            general_copy(v_buf, indices, v_home, indices, self.policy.compress_weight)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None, self.policy.compress_weight)
        general_copy(v_home, indices, v_new, None, self.policy.compress_weight)

    def forward(self, hidden, cache_read_buf=None, cache_write_buf=None, weight_read_buf=None, attention_mask=None, i=0, k=0):
        print(f"[FLEX_LlamaAttention] Input hidden shape: {hidden.val.shape}")
        print(f"[FLEX_LlamaAttention] Input hidden device: {hidden.val.device}")
        print(f"[FLEX_LlamaAttention] Input hidden dtype: {hidden.val.dtype}")
        hidden = hidden.val.data
        # import pdb; pdb.set_trace()
        print(f"[FLEX_LlamaAttention] Input hidden data: {hidden}")
        # Get weight tensors
        if weight_read_buf is None:
            weight_read_buf = self.weight_read_buf
        w_q, w_k, w_v, w_out, input_layernorm, rotary_embed = weight_read_buf.val
        if isinstance(w_q, tuple):
            w_q = w_q[0].data
            w_k = w_k[0].data
            w_v = w_v[0].data
            w_out = w_out[0].data
            input_layernorm = input_layernorm[0].data
            rotary_embed = rotary_embed[0].data
        # Ensure hidden states has the correct shape
        if hidden.shape[-1] != self.config.hidden_size:
            print(f"[FLEX_LlamaAttention] Shape mismatch: expected {self.config.hidden_size} but got {hidden.shape[-1]}")
            # If the hidden size is half of expected, duplicate the data
            if hidden.shape[-1] * 2 == self.config.hidden_size:
                print("[FLEX_LlamaAttention] Duplicating data to match expected size")
                hidden = torch.cat([hidden, hidden], dim=-1)
            else:
                raise ValueError(f"Hidden states dimension mismatch: expected {self.config.hidden_size} but got {hidden.shape[-1]}")

        # Initialize donate list for memory management
        donate = [False] * 14  # Create a list of 14 False values for donate flags

        if i == 0:
            # prefill
            mask, donate[1] = attention_mask.val.data if hasattr(attention_mask.val, 'data') else attention_mask.val, True
            h, new_k_cache, new_v_cache = self.compute.mha_llama(hidden, mask, w_q, w_k, w_v, w_out,
                                       self.num_heads, donate, self.policy.compress_cache, self.policy.comp_cache_config, input_layernorm, rotary_embed, compress_weight=self.policy.compress_weight)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:
            # decoding
            mask, donate[1] = attention_mask.val.data if hasattr(attention_mask.val, 'data') else attention_mask.val, True
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            
            # Access .data attribute for cache tensors if available
            k_cache = k_cache.data if hasattr(k_cache, 'data') else k_cache
            v_cache = v_cache.data if hasattr(v_cache, 'data') else v_cache
            
            # Extract the actual tensors from the tuples if they are tuples
            if isinstance(k_cache, tuple):
                k_cache = k_cache[0]
            if isinstance(v_cache, tuple):
                v_cache = v_cache[0]
                
            # 确保缓存张量也是 NF4 格式
            if hasattr(self.compute, 'compressed_device') and self.compute.compressed_device is not None:
                # 检查张量是否有 compress 方法
                if hasattr(k_cache, 'compress'):
                    k_cache = k_cache.compress(self.policy.comp_cache_config)
                if hasattr(v_cache, 'compress'):
                    v_cache = v_cache.compress(self.policy.comp_cache_config)
            
            # If the weights are compressed, decompress them
            
            h, new_k_cache, new_v_cache = self.compute.mha_gen_llama(
                hidden, mask, w_q,
                w_k, w_v, w_out, self.num_heads,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config,
                input_layernorm,
                rotary_embed, compress_weight=self.policy.compress_weight)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h
