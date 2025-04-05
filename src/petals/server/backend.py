from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import PretrainedConfig

from petals.data_structures import InferenceMetadata
from petals.server.memory_cache import MemoryCache
from petals.server.task_pool import PrioritizedTaskPool
from petals.utils.misc import get_size_in_bytes, is_dummy

logger = get_logger(__name__)
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

class TransformerBackend(ModuleBackend): # hivemind: ModuleBackend.module: nn.Module
    """A wrapper for a transformer block that can process requests for forward, backward and inference"""

    _peft_module = None

    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        memory_cache: MemoryCache,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        import petals.utils.peft as _peft_module

        self._peft_module = _peft_module

        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.memory_cache = memory_cache
        self.max_chunk_size_bytes = max_chunk_size_bytes

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )  # note: inference_pools may be merged later, see merge_inference_pools_inplace
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        for shard in self.module.module_shards:
            for submodule in shard.modules():
                if isinstance(submodule, config.attn_class):
                    self.shard_num_heads.append(submodule.num_heads)
        assert len(self.shard_num_heads) == len(self.module.devices)
        assert sum(self.shard_num_heads) == config.num_attention_heads

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * get_size_in_bytes(descr.dtype)

        # 创建 CPU 设备列表
        num_cpus = 1  # 可以根据需要调整
        cpus = [torch.device('cpu') for _ in range(num_cpus)]
        
        # 设置 TensorParallel 模块使用 CPU 设备
        self.module.devices = cpus
        
        # 如果模块有 module_shards，将它们移动到 CPU
        if hasattr(self.module, 'module_shards'):
            for shard in self.module.module_shards:
                shard.to('cpu')
        
        # 设置输出设备为 CPU
        if hasattr(self.module, 'output_device_index'):
            self.module.output_device_index = 0  # 使用第一个 CPU 作为输出设备
        
        # 标记需要延迟初始化
        self.module.need_delayed_init = True
        
        # 记录原始设备，以便在需要时恢复
        self.original_devices = self.module.devices
        self.original_output_device_index = self.module.output_device_index

    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache_tensors = []
        for device, num_heads in zip(self.module.devices, self.shard_num_heads):
            num_heads //= self.config.num_key_value_groups
            if hasattr(self.config, "num_key_value_heads"):
                num_heads = self.config.num_key_value_heads
            keys = TensorDescriptor((batch_size, num_heads, head_dim, max_length), dtype=self.dtype, device=device)
            values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.extend((keys, values))
        return cache_tensors

    def forward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            # 在 forward 之前，确保模型在正确的设备上
            self._ensure_model_on_device()
            return super().forward(*inputs)

    def backward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            # 在 backward 之前，确保模型在正确的设备上
            self._ensure_model_on_device()
            return super().backward(*inputs)

    def _ensure_model_on_device(self):
        """确保模型在正确的设备上，如果需要，从 CPU 加载到 GPU"""
        # 检查当前设备是否与原始设备不同
        if self.module.devices != self.original_devices:
            # 将模型移动到原始设备
            self.module.devices = self.original_devices
            self.module.output_device_index = self.original_output_device_index
            
            # 如果模块有 module_shards，将它们移动到原始设备
            if hasattr(self.module, 'module_shards'):
                for shard, device in zip(self.module.module_shards, self.original_devices):
                    shard.to(device)
            
            # 标记需要延迟初始化
            self.module.need_delayed_init = True

    @torch.inference_mode() # 进入推理模式，不计算梯度，从而节省内存 
    def inference_step( # 每一个block都会执行一次, 
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量 
        hypo_ids: torch.LongTensor,  # 假设的 ID 
        inference_info: InferenceMetadata,  # 推理相关元数据
    ) -> Tuple[torch.Tensor, ...]:
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]" # 确保隐藏状态是三维的 
        seq_len = hidden_states.shape[1] # 获取序列的长度 
        # print("transformer backend inference step : seq_len", seq_len)
        see_memory_usage("transformer backend inference step : seq_len")
        
        # 在推理之前，确保模型在正确的设备上
        self._ensure_model_on_device()
        
        with self.memory_cache.use_cache(
            *inference_info.cache_handles  # 使用缓存，降低内存需求  
        ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter): # 使用adapter进行推理  
            self._reorder_cache_inplace(cache_tensors, hypo_ids) # 根据假设 ID 重新排列缓存  

            # We chunk the inputs so that peak memory for long sequences fits into `autograd_memory`
            # reserved in `Server._choose_num_blocks()`. This saves us from OOMs if `max_chunk_size_bytes`
            # is at least 4-6x less than `autograd_memory`.
            max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info) # 估计最大分块长度 
            print("transformer backend inference step() : max_chunk_length", max_chunk_length)
            see_memory_usage("transformer backend inference step : seq_len")
            output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None # 初始化输出状态
            # print("transformer backend inference step : output_hidden_states", output_hidden_states) # output_hidden_states:None
            layer_past = self._select_layer_past(cache_tensors, inference_info.prefix_length) # 选择上一个层的缓存状态 
            for offset in range(0, seq_len, max_chunk_length): # 遍历序列以按块处理隐藏状态   only run offset=0
                hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :] # 获取当前的隐藏状态块 
                print('transformer backend inference step() offset ', offset )
                print('transformer backend inference step() offset + max_chunk_length',  (offset + max_chunk_length))
                # output_hidden_states_chunk, new_kvs = self.module.forward(
                #     hidden_states_chunk, layer_past=layer_past, use_cache=True # 前向传播，返回新的键值状态  
                # )
                import pdb;pdb.set_trace()
                see_memory_usage("----before -transformer backend inference step output_hidden_states_chunk,= self.module.forward(")
                output_hidden_states_chunk,= self.module.forward(
                    hidden_states_chunk, layer_past=layer_past, use_cache=False # 前向传播，返回新的键值状态  
                )
                see_memory_usage("----after -transformer backend inference step output_hidden_states_chunk,= self.module.forward(")
                
                if seq_len > max_chunk_length:
                    output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk # 存储输出
                else:
                    output_hidden_states = output_hidden_states_chunk  # saves one memcopy # 仅复制一次内存
                # layer_past = new_kvs # 更新缓存状态

            # self._update_cache_inplace(cache_tensors, new_kvs, inference_info.prefix_length) # 更新缓存 
            # import pdb; pdb.set_trace()
            print('backend.py output_hidden_states.shape ', output_hidden_states.shape)
            return (output_hidden_states,) # 返回输出的隐藏状态

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory, given that
        # the model uses multi-query attention
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids

    def _select_layer_past(self, cache_tensors: Sequence[torch.Tensor], prefix_length: int) -> Sequence[torch.Tensor]:
        """Extract first {prefix_length} tokens and reshape them such that they can be used as layer_past"""
        key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])
        for i in range(len(key_cache)):
            key_cache[i] = key_cache[i].flatten(0, 1)[:, :, :prefix_length]
            # shape: [batch * num_kv_heads, head_dim, kv_length]
            value_cache[i] = value_cache[i].flatten(0, 1)[:, :prefix_length]
            # shape: [batch * num_kv_heads, kv_length, head_dim]
        layer_past = tuple(chain(*zip(key_cache, value_cache)))
        return PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past

    def _update_cache_inplace(
        self, cache_tensors: Sequence[torch.Tensor], new_kvs: Sequence[torch.Tensor], prefix_length: int
    ):
        """Writes new key/value tensors back into cache, works in-place"""
        _batch_size_times_num_kv_heads, head_dim, new_length = new_kvs[0].shape
        for cache_key, new_key in zip(cache_tensors[0::2], new_kvs[0::2]):
            new_key = new_key.view(*cache_key.shape[:3], new_length)
            cache_key[:, :, :, prefix_length:new_length] = new_key[:, :, :, prefix_length:new_length]
        for cache_value, new_value in zip(cache_tensors[1::2], new_kvs[1::2]):
            new_value = new_value.view(*cache_value.shape[:2], new_length, head_dim)
            cache_value[:, :, prefix_length:new_length, :] = new_value[:, :, prefix_length:new_length, :]

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy


def merge_inference_pools_inplace(backends: Dict[ExpertUID, TransformerBackend]):
    """Replace each backend's rpc_inference pools with a combined pool runs multiple blocks in one call"""
    assert len(backends) != 0 and all(isinstance(b, TransformerBackend) for b in backends.values())
    print('............... come into the merge_inference_pools_inplace() ' )
    first_pool = next(iter(backends.values())).inference_pool
    merged_pool = PrioritizedTaskPool(
        _MergedInferenceStep(backends),
        max_batch_size=first_pool.max_batch_size,
        device=first_pool.device,
        name=f"merged_inference",
    )
    for backend in backends.values():
        assert not backend.inference_pool.is_alive()
        backend.inference_pool = merged_pool
        # here, the backend is "blocks" in the server.py line 536

class _MergedInferenceStep:
    def __init__(self, backends: Dict[ExpertUID, TransformerBackend]):
        self.backends = backends

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_infos: Sequence[InferenceMetadata],
        *optional_prompts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        assert len(inference_infos) == len(
            optional_prompts
        ), f"found {len(inference_infos)} blocks but {len(optional_prompts)} prompts"
        print('............... come into the _MergedInferenceStep __call__' )
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            print('............... come into the _MergedInferenceStep __call__ inference_info.uid ', inference_info.uid)
            (hidden_states,) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info)
        # import pdb; pdb.set_trace()
        return (hidden_states,)
