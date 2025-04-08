"""
LLaMA intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
See commit history for authorship.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (
    # LlamaAttention,
    LlamaConfig,
    # LlamaDecoderLayer,
    # LlamaMLP,
    LlamaRMSNorm,
    repeat_kv,
    rotate_half,
)

import numpy as np
from petals.utils.cuda_graphs import make_inference_graphed_callable
from petals.flexgen_utils.ExecutionEnv import ExecutionEnv
from petals.flexgen_utils.compression import CompressionConfig
from petals.flexgen_utils.policy import Policy
from petals.flexgen_utils.torch_tensor import TorchTensor
from petals.flexgen_utils.torch_device import TorchDevice
from petals.flexgen_utils.utils import ValueHolder, array_1d, array_2d, array_3d
from petals.models.llama.flex_llama import FLEX_LlamaAttention, FLEX_LlamaMLP, LlamaDecoderLayer, DUMMY_WEIGHT
from petals.flexgen_utils.llama_config import get_llama_config, download_llama_weights
from petals.flexgen_utils.task import Task
from transformers import AutoTokenizer
import os

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


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed




class FLEX_LlamaRMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps=1e-6)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"





class OptimizedLlamaAttention(FLEX_LlamaAttention):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)
    # def __init__(self, *args, env, policy, layer_id, **kwargs):
    #     super().__init__(*args, env, policy, layer_id, **kwargs)
        self._rotary_graph = None
        self.temp_hidden_states = ValueHolder()
        
        # self.env = env
        # self.layer_id = layer_id
        # self.policy = policy
        # self.compute = self.env.gpu

        # self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
        #                         else self.compute)
        # self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
        #                           else self.env.gpu)
        
        # self.task = None
        

    def _optimized_apply_rotary(self, query_states, key_states, cos, sin):
        if self._rotary_graph is None:
            self._rotary_graph = make_inference_graphed_callable(
                apply_rotary_pos_emb, sample_args=(query_states, key_states, cos, sin)
            )
        return self._rotary_graph(query_states, key_states, cos, sin)

    def forward(
        self,
        hidden_states,  # 可以是ValueHolder或torch.Tensor
        cache_read_buf: ValueHolder = None,
        weight_read_buf: ValueHolder = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_write_buf: ValueHolder = None,
        position_ids: Optional[torch.LongTensor] = None,
        k: int = 0,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False
        assert not output_attentions
        see_memory_usage("-----------------------------------------enter llama attention forward ")
        
        # 确保hidden_states是ValueHolder类型
        if not isinstance(hidden_states, ValueHolder):
            # 如果是torch.Tensor，转换为ValueHolder
            # Get the device string from the torch.device object
            device_str = str(hidden_states.device)
            tensor_data = TorchTensor(shape=hidden_states.shape, data=hidden_states, dtype=hidden_states.dtype, device=TorchDevice(device_str))
            hidden_holder = ValueHolder()
            hidden_holder.store(tensor_data)
            hidden_states = hidden_holder
        
        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, 
                past_seen_tokens + hidden_states.val.data.shape[1],
                device=hidden_states.val.data.device
            ).unsqueeze(0)
        
        print('block.py : class OptimizedLlamaAttention forward(): position_ids,', position_ids)
        see_memory_usage("-----------------------------------------after position_ids ")
        i = int(position_ids.item())
        
        # 调用父类的forward方法，确保参数顺序正确
        # 注意：FLEX_LlamaAttention的forward方法不接受cache_write_buf参数
        # 我们需要在内部处理这个参数
        if cache_write_buf is not None:
            # 如果提供了cache_write_buf，我们需要在内部处理它
            # 这里我们可以在调用父类的forward方法后，将结果存储到cache_write_buf中
            result = super(OptimizedLlamaAttention, self).forward(
                hidden=hidden_states,
                cache_read_buf=cache_read_buf,
                weight_read_buf=weight_read_buf,
                attention_mask=attention_mask,
                i=i,
                k=k
            )
            # 假设result是一个元组，其中第二个元素是cache_write_buf需要的数据
            if isinstance(result, tuple) and len(result) > 1:
                cache_write_buf.store(result[1])
            return result
        else:
            # 如果没有提供cache_write_buf，直接调用父类的forward方法
            result = super(OptimizedLlamaAttention, self).forward(
                hidden=hidden_states,
                cache_read_buf=cache_read_buf,
                weight_read_buf=weight_read_buf,
                attention_mask=attention_mask,
                i=i,
                k=k
            )
            # 更新temp_hidden_states
            if isinstance(result, tuple):
                self.temp_hidden_states.val = result[0]
            else:
                self.temp_hidden_states.val = result
            return self.temp_hidden_states.val, None, None


class OptimizedLlamaDecoderLayer(LlamaDecoderLayer):  # used in block_utils.py return config.block_class(config)
    def __init__(self, config: LlamaConfig,layer_id: int, env: ExecutionEnv, policy: Policy, weight_home: array_1d, path: str, ):
        see_memory_usage("-----------------------------------------OptimizedLlamaDecoderLayer  init ")
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        # print('OptimizedLlamaDecoderLayer config ', config)
        self.config = config
        # self.devices = (device(type='cuda', index=0),)
        
        self.num_heads = config.num_attention_heads
        # self.self_attn = OptimizedLlamaAttention(config=config, layer_idx=0)
        # self.mlp = LlamaMLP(config=config)
        ########---------------------------------------------
        self.self_attn = OptimizedLlamaAttention(config=config, env=env, policy=policy, layer_id=self.layer_id )
        #layer_idx only matters for KV caching, and we re-implement it in Petals
        self.mlp = FLEX_LlamaMLP(config=config, env=env, policy=policy,layer_id=self.layer_id )
         ########---------------------------------------------
        self.input_layernorm = FLEX_LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FLEX_LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_attn_graph = None
        self.post_attn_graph = None
        
        
        
        self.llama_config = get_llama_config('huggyllama/llama-7b')
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        
        layers = []

        # layers.append(InputEmbed(self.llama_config, self.env, self.policy))
        layers.append(self.self_attn)
        layers.append(self.mlp)
        # layers.append(self.input_layernorm)
        # layers.append(self.post_attention_layernorm)

        self.layers = layers
        self.num_layers = len(layers)
        # self.num_layers = 1 # current block is only one decoder layer
        # print('block.py, class OptimizedLlamaDecoderLayer(LlamaDecoderLayer): self.mlp ', self.mlp)
        # dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
        # dev_choices = [env.disk, env.cpu, env.gpu]
        
        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()
        # see_memory_usage("-----------------------------------------before cuda stream init ")
        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()
        
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j] current block is decoder layer j, contains 4 layers
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # self.weight_read_buf = ValueHolder()
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        # print('before init_all_weights OptimizedLlamaDecoderLayer self.config', self.config)
        see_memory_usage("-----------------------------------------before init_all_weights ")
        # Initialize weights and apply final processing
        self.init_all_weights() #-------------------------************
        # print('OptimizedLlamaDecoderLayer self.config', self.config)
        see_memory_usage("-----------------------------------------after init_all_weights ")
        
        self.temp_hidden = ValueHolder() ######
        
    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)
        
    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)
            
    def init_weight(self, j):
        # print('self.llama_config ', self.llama_config)
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.llama_config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        see_memory_usage("----------------------------------before download_llama_weights in init_weights ")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_llama_weights(self.llama_config.name, self.path)
        see_memory_usage(str(j)+" layer----------------------------------before self.layers[j].init_weight(self.weight_home[j], expanded_path) ")
        
        self.layers[j].init_weight(self.weight_home[j], expanded_path)
        see_memory_usage(str(j)+" layer----------------------------------after self.layers[j].init_weight(self.weight_home[j], expanded_path) ")
        
        
    
    
    
    
    def _optimized_input_layernorm(self, hidden_states):
        if self.pre_attn_graph is None:
            self.pre_attn_graph = make_inference_graphed_callable(
                self.input_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.pre_attn_graph(hidden_states)

    def _optimized_output_layernorm(self, hidden_states):
        if self.post_attn_graph is None:
            self.post_attn_graph = make_inference_graphed_callable(
                self.post_attention_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.post_attn_graph(hidden_states)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        # input_ids = self.output_ids[left:right, :self.task.prompt_len]#####

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        # val.load_from_np((input_ids != self.config.pad_token_id)) #######
        self.attention_mask[k].store(val)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        max_new_tokens: int=1, ############
        do_sample: bool=True, ############
        temperature: float=0.6, ############
        stop: Optional[int] = None, ############
        debug_mode: Optional[str] = None, ############
        cut_gen_len: Optional[int] = None, ############
        top_p: float = 0.9, ############
        verbose: int = 0, ############
        # k: int, ######## the num_gpu_batches 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if isinstance(hidden_states, ValueHolder):
            hidden_states = hidden_states.val.data
        if isinstance(hidden_states, TorchTensor):
            hidden_states = hidden_states.data
        see_memory_usage("-----------------------------------------before optimized llama decoder layer forward ")
        print(f"[OptimizedLlamaDecoderLayer] Input hidden_states shape: {hidden_states.shape}")
        print(f"[OptimizedLlamaDecoderLayer] Input hidden_states device: {hidden_states.device}")
        print(f"[OptimizedLlamaDecoderLayer] Input hidden_states dtype: {hidden_states.dtype}")
        
        # Ensure hidden states has the correct shape
        if hidden_states.shape[-1] != self.config.hidden_size:
            print(f"[OptimizedLlamaDecoderLayer] Shape mismatch: expected {self.config.hidden_size} but got {hidden_states.shape[-1]}")
            # If the hidden size is half of expected, duplicate the data
            if hidden_states.shape[-1] * 2 == self.config.hidden_size:
                print("[OptimizedLlamaDecoderLayer] Duplicating data to match expected size")
                hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
            else:
                raise ValueError(f"Hidden states dimension mismatch: expected {self.config.hidden_size} but got {hidden_states.shape[-1]}")

        # 将hidden_states转换为ValueHolder类型
        # Get the device string from the torch.device object
        device_str = str(hidden_states.device)
        tensor_data = TorchTensor(shape=hidden_states.shape, data=hidden_states, dtype=hidden_states.dtype, device=TorchDevice(device_str))
        hidden_holder = ValueHolder()
        hidden_holder.store(tensor_data)
        
        # 初始化hidden属性
        self.hidden[0][0][0].store(tensor_data)
        
        residual = hidden_states
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", padding_side="left", legacy=False)
        tokenizer.pad_token = '[PAD]'
        num_prompts = 1
        
        see_memory_usage("-----------------------------------------before cuda stream init ")
        prompt_len, gen_len, cut_gen_len = 1,1,1 ##########-------------------------------------
        inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
        print('inputs , ', inputs)
       
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
        num_prompts = len(task.inputs)
        prompt_len, gen_len = task.prompt_len, task.gen_len
        
        self.output_ids = np.ones((num_prompts, prompt_len + gen_len), dtype=np.int64)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        
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

        data = hidden_states
        device = TorchDevice(data.device)
        tensor_data = TorchTensor(shape=data.shape, data=data, dtype=data.dtype, device=device)
        self.hidden[0][0][0].store(tensor_data)
        
        self.task = task
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)
        
        debug_mode = None #######
        overlap = False #######
        if debug_mode is None:
            if not overlap:
                i = 0 ############# to simplify the woekflow, we only consider the one token each time 
                for k in range(self.num_gpu_batches):
                    self.update_attention_mask(i, k)
                
                for j in range(self.num_layers):
                    for k in range(self.num_gpu_batches):
                        self.load_weight(i, j, k, overlap=False)
                    
                    for k in range(self.num_gpu_batches):
                        self.load_cache(i, j, k, overlap=False)
                         
                        see_memory_usage('-----------------------------------------before compute_layer '+ str(i)+ '' + str(j)+' '+str(k))
                        hidden_states = self.compute_layer(i, j, k).data.clone()  
                        see_memory_usage('-----------------------------------------after compute_layer '+ str(i)+ '' + str(j)+' '+str(k))
                
        outputs = (hidden_states,)
        torch.cuda.empty_cache()  
        return outputs
    
    def get_shape_3d(self, lst):  
        if not lst:  # Check if the outer list is empty  
            return (0, 0, 0)  

        depth = len(lst)  
        num_rows = len(lst[0]) if lst[0] else 0  # Length of the first inner list  
        num_cols = len(lst[0][0]) if lst[0] and lst[0][0] else 0  # Length of the first innermost list  
        return (depth, num_rows, num_cols) 
    
    def set_task(self, task):
        self.self_attn.set_task(task)
        self.mlp.set_task(task)
    #######################################################################################
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
        # 
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
            val = self.hidden[i][j - 1][k].pop().move(dst)
        
        # self.hidden[i][j][k].val = None  # 重置
        self.hidden[i][j][k].store(val)

    def load_hidden_mlp(self, i, j, k):
        self.hidden[i][j][k].store(self.temp_hidden.val)

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
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
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
                x.val = x.val.move(self.act_home)
                
    def compute_layer(self, i, j, k):
        # print('compute_layer', self.hidden[i][j][k])
        # print('i, j, k  '+ str(i)+','+str(j)+','+str(k))
        if j == 1:
            self.hidden[i][j][k].val = self.temp_hidden.val
        # print('compute_layer hidden val.data.shape', self.hidden[i][j][k].val.data.shape)
        print(self.hidden[i][j][k].val)
        # print('token i', i)
        pos_id = torch.tensor(i)
        # print('pos_id i', pos_id)
        # import pdb;pdb.set_trace()
        # print('layer j ', j)
        
        # Profile memory before forward pass
        see_memory_usage(f"-----------------------------------------before compute_layer {j} forward pass for token {i}")
        
        # 调用forward方法，确保参数顺序正确
        # 注意：self.hidden[i][j][k]是ValueHolder类型，不需要转换为TorchTensor
        result = self.layers[j].forward(
            hidden_states=self.hidden[i][j][k], 
            cache_read_buf=self.cache_read_buf[j][k],
            weight_read_buf=self.weight_read_buf[j], 
            attention_mask=self.attention_mask[k],
            cache_write_buf=self.cache_write_buf[j][k],  # 传递cache_write_buf参数
            position_ids=pos_id,
            k=k
        )
        
        # 如果result是一个元组，我们需要提取hidden_states
        if isinstance(result, tuple):
            hidden_states = result[0]
        else:
            hidden_states = result
            
        # 更新temp_hidden
        self.temp_hidden.val = hidden_states
        
        # Profile memory after forward pass
        see_memory_usage(f"-----------------------------------------after compute_layer {j} forward pass for token {i}")
        
        print('self.temp_hidden.val.data ', self.temp_hidden.val.data)
        print('self.temp_hidden.val.data.shape ', self.temp_hidden.val.data.shape)
        return self.temp_hidden.val
    
#######################################################################################

class WrappedLlamaBlock(OptimizedLlamaDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化hidden属性，与OptimizedLlamaDecoderLayer中的初始化方式相同
        gen_len = 1  # 默认值，与OptimizedLlamaDecoderLayer中的默认值相同
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past
        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            past_key_value = self._reorder_cache_from_bloom_to_llama(past_key_value, batch_size, past_key_values_length)

        assert position_ids is None

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
        )

        # 将hidden_states转换为ValueHolder类型
        # Get the device string from the torch.device object
        device_str = str(hidden_states.device)
        tensor_data = TorchTensor(shape=hidden_states.shape, data=hidden_states, dtype=hidden_states.dtype, device=TorchDevice(device_str))
        hidden_holder = ValueHolder()
        hidden_holder.store(tensor_data)
        
        # 初始化hidden属性
        self.hidden[0][0][0].store(tensor_data)

        outputs = super().forward( ############
            hidden_states=hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        print('block.py WrappedLlamaBlock forward : outputs ', outputs)
        print('use_cache', use_cache)
        # use_cache
        # if use_cache:
        #     present_key_value = outputs[-1]
        #     present_key_value = self._reorder_cache_from_llama_to_bloom(
        #         present_key_value, batch_size, seq_length_with_past
        #     )
        #     outputs = outputs[:-1] + (present_key_value,)
        
        return outputs

    def _reorder_cache_from_bloom_to_llama(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_from_llama_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        value_states = value_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.view(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = [
        # "Simply put, the theory of relativity states that ",

        # "I believe the meaning of life is",
        "",
        # """Translate English to French:
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
    ]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


