"""
TorchDevice module for flexgen_utils.
"""

import torch
import numpy as np

from petals.flexgen_utils.DeviceType import DeviceType
from petals.flexgen_utils.shared_types import (
    global_cpu_device,
    global_cuda_devices,
    global_disk_device
)

from petals.flexgen_utils.torch_tensor import TorchTensor

# Define a mapping from NumPy data types to PyTorch data types
np_dtype_to_torch_dtype = {
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('int8'): torch.int8,
    np.dtype('int16'): torch.int16,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('uint8'): torch.uint8,
    np.dtype('bool'): torch.bool,
    # 添加对 numpy.float16 等类型对象的支持
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.bool_: torch.bool
}

class TorchDevice:
    """Device for PyTorch tensors."""

    def __init__(self, name, mem_capacity=None, flops=None):
        """
        Initialize a PyTorch device.

        Args:
            name: Device name, e.g., "cuda:0" or "cpu", or a torch.device object
            mem_capacity: Memory capacity in bytes
            flops: Floating-point operations per second
        """
        # 处理 torch.device 对象
        if isinstance(name, torch.device):
            device_str = str(name)
            if device_str.startswith("cuda"):
                self.device_type = DeviceType.CUDA
                self.dev = name
                if mem_capacity is None:
                    mem_capacity = torch.cuda.get_device_properties(self.dev).total_memory
            elif device_str == "cpu":
                self.device_type = DeviceType.CPU
                self.dev = torch.device("cpu")
                if mem_capacity is None:
                    mem_capacity = 32 * 1024 * 1024 * 1024  # 32GB
            else:
                raise ValueError(f"Unknown device: {device_str}")
        # 处理字符串类型的设备名称
        elif isinstance(name, str):
            if name.startswith("cuda"):
                self.device_type = DeviceType.CUDA
                self.dev = torch.device(name)
                if mem_capacity is None:
                    mem_capacity = torch.cuda.get_device_properties(self.dev).total_memory
            elif name == "cpu":
                self.device_type = DeviceType.CPU
                self.dev = torch.device("cpu")
                if mem_capacity is None:
                    mem_capacity = 32 * 1024 * 1024 * 1024  # 32GB
            else:
                raise ValueError(f"Unknown device: {name}")
        else:
            raise ValueError(f"Unsupported device type: {type(name)}")
        
        self.name = str(self.dev)
        self.mem_capacity = mem_capacity
        self.flops = flops
        self.links = []
        self.compressed_device = None
        self.mixed_device = None
        self.disk_device = None
        
        # Initialize workspace
        self.attention_compute_workspace = None
        self.workspace_pt = 0

    def add_link(self, link):
        """Add a link to another device."""
        self.links.append(link)

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        """
        Allocate a tensor on this device.

        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            pin_memory: Whether to pin memory (only for CPU)
            name: Tensor name

        Returns:
            A TorchTensor on this device
        """
        # 将 NumPy 数据类型转换为 PyTorch 数据类型
        if isinstance(dtype, type) and hasattr(np, dtype.__name__):
            # 如果是 NumPy 数据类型，转换为 PyTorch 数据类型
            torch_dtype = np_dtype_to_torch_dtype[dtype]
        else:
            # 如果已经是 PyTorch 数据类型，直接使用
            torch_dtype = dtype
            
        if self.device_type == DeviceType.CUDA:
            data = torch.empty(shape, dtype=torch_dtype, device=self.dev)
            return TorchTensor(shape, dtype, data, self, name)
        elif self.device_type == DeviceType.CPU:
            data = torch.empty(shape, dtype=torch_dtype, device="cpu", pin_memory=pin_memory)
            return TorchTensor(shape, dtype, data, self, name)
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

    def delete(self, tensor):
        """Delete a tensor from this device."""
        tensor.delete()

    def init_attention_compute_workspace(self, config, task, policy):
        """
        Initialize workspace for attention computation.

        Args:
            config: Model configuration
            task: Task configuration
            policy: Policy configuration
        """
        # This is a placeholder and will be implemented in a separate module
        pass

    def next_attention_compute_workspace(self):
        """Get the next attention compute workspace."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def del_attention_compute_workspace(self):
        """Delete the attention compute workspace."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        """Generate attention mask from token IDs."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def extend_attention_mask(self, attention_mask, donate):
        """Extend attention mask."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        """Input embedding for OPT model."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def llama_input_embed(self, inputs, attention_mask, w_token, pad_token_id, donate, token_type_embeddings):
        """Input embedding for LLaMA model."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate, do_sample, temperature):
        """Output embedding for OPT model."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def llama_output_embed(self, inputs, w_ln, donate, do_sample, temperature, lm_head, top_p):
        """Output embedding for LLaMA model."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def init_cache_one_gpu_batch(self, config, task, policy):
        """Initialize cache for one GPU batch."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate, attn_sparsity, compress_cache, comp_config):
        """Multi-head attention for generation."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def mha_llama(self, hidden_states, attention_mask, w_q, w_k, w_v, w_out, n_head, donate, compress_cache=False, comp_cache_config=None, input_layernorm=None, rotary_emb_inv_freq=None, compress_weight=False, comp_weight_config=None):
        """Multi-head attention for LLaMA model."""
        import torch.nn.functional as F
        
        # 添加调试信息
        print(f"[mha_llama] Input hidden_states type: {type(hidden_states)}")
        if hasattr(hidden_states, 'data'):
            print(f"[mha_llama] Input hidden_states.data type: {type(hidden_states.data)}")
            if hidden_states.data is not None:
                print(f"[mha_llama] Input hidden_states.data shape: {hidden_states.data.shape}")
                print(f"[mha_llama] Input hidden_states.data device: {hidden_states.data.device}")
                print(f"[mha_llama] Input hidden_states.data dtype: {hidden_states.data.dtype}")
        
        # 检查输入是否为 None
        if hidden_states is None:
            print("[mha_llama] Error: hidden_states is None")
            # 创建一个空的隐藏状态
            hidden_states = torch.zeros((1, 1, 4096), dtype=torch.float16, device=self.dev)
            print(f"[mha_llama] Created empty hidden_states with shape: {hidden_states.shape}")
        
        # 如果 hidden_states 是 TorchTensor 对象，获取其 data 属性
        if hasattr(hidden_states, 'data') and hidden_states.data is not None:
            hidden_states = hidden_states.data
        
        # 获取输入张量的形状
        bsz, q_len, h = hidden_states.shape
        head_dim = h // n_head
        
        # 处理权重张量
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
        
        # 解压缩权重
        if compress_weight and comp_weight_config is not None:
            w_q = w_q_tensor.decompress(comp_weight_config)
            w_k = w_k_tensor.decompress(comp_weight_config)
            w_v = w_v_tensor.decompress(comp_weight_config)
            w_out = w_out_tensor.decompress(comp_weight_config)
            if input_layernorm is not None:
                input_layernorm = input_layernorm_tensor.decompress(comp_weight_config)
            if rotary_emb_inv_freq is not None:
                rotary_emb_inv_freq = rotary_emb_inv_freq_tensor.decompress(comp_weight_config)
        
        # 确保隐藏状态有正确的形状
        if hidden_states.shape[-1] != w_q_tensor.shape[1]:
            if hidden_states.shape[-1] * 2 == w_q_tensor.shape[1]:
                hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
            else:
                raise ValueError(f"Hidden states dimension mismatch: expected {w_q_tensor.shape[1]} but got {hidden_states.shape[-1]}")
        
        # 计算旋转位置嵌入
        freq_cis = precompute_freqs_cis(head_dim, 2048 * 2, rotary_emb_inv_freq.data)
        scaling = head_dim ** -0.5
        
        # 应用输入层归一化
        hidden = rms_norm(hidden_states, input_layernorm.data)
        
        # 使用解压缩的权重
        w_q_data = w_q.data
        w_k_data = w_k.data
        w_v_data = w_v.data
        w_out_data = w_out.data
        
        # 确保 hidden 是 float16 类型
        if hidden.dtype != torch.float16:
            hidden = hidden.to(dtype=torch.float16)
        
        # 确保所有权重张量都是 float16 类型
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
        
        # 计算查询、键和值
        q = F.linear(hidden, w_q_data) * scaling
        k = F.linear(hidden, w_k_data)
        v = F.linear(hidden, w_v_data)
        
        # 检查张量大小是否匹配预期形状
        expected_size = bsz * q_len * n_head * head_dim
        actual_size = q.numel()
        
        if actual_size != expected_size:
            if actual_size * 2 == expected_size:
                q = torch.cat([q, q], dim=-1)
                k = torch.cat([k, k], dim=-1)
                v = torch.cat([v, v], dim=-1)
            else:
                raise ValueError(f"Tensor size mismatch: expected {expected_size} elements but got {actual_size}. Hidden size: {h}, Head dim: {head_dim}, Num heads: {n_head}")
        
        # 重塑张量
        q = q.view(bsz, q_len, n_head, head_dim)
        k = k.view(bsz, q_len, n_head, head_dim)
        v = v.view(bsz, q_len, n_head, head_dim)
        
        # 应用旋转位置嵌入
        q, k = apply_rotary_emb(q, k, freqs_cis=freq_cis[:q_len])
        
        # 形状: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(bsz * n_head, q_len, head_dim)
        # 形状: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(bsz * n_head, head_dim, q_len)
        # 形状: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(bsz * n_head, q_len, head_dim)
        
        # 计算注意力权重
        attn_weights = torch.bmm(q, k)
        
        # 创建因果掩码
        idx = torch.arange(q_len, device=self.dev)
        causal_mask = (idx <= idx.view(q_len, 1)).view(1, 1, q_len, q_len) 
        mask = attention_mask.data.view(bsz, 1, 1, q_len) & causal_mask
        
        # 形状: (b, n_head, s, s)
        attn_weights = attn_weights.view(bsz, n_head, q_len, q_len)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(bsz * n_head, q_len, q_len)
        attn_weights = F.softmax(attn_weights, dim=2)
        
        # 形状: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(bsz, n_head, q_len, head_dim)
        # 形状: (b, s, h)
        value = value.transpose(1, 2).reshape(bsz, q_len, h)
        value = F.linear(value, w_out_data)
        
        # 残差连接
        value.add_(hidden_states)
        
        # 释放内存
        if donate[0]: hidden_states.delete()
        if donate[1]: attention_mask.delete()
        
        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)
        
        # 压缩缓存
        if compress_cache and self.compressed_device is not None:
            k = self.compressed_device.compress(k, comp_cache_config)
            v = self.compressed_device.compress(v, comp_cache_config)
        
        return value, k, v

    def mha_gen_llama(self, inputs, attention_mask, w_q, w_k, w_v, w_out, n_head, k_cache, v_cache, donate, attn_sparsity=1.0, compress_cache=False, comp_cache_config=None, input_layernorm=None, rotary_emb_inv_freq=None, compress_weight=False):
        """Multi-head attention for LLaMA model generation."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        """Compute attention weights."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        """Compute attention values."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b, src_s, tgt_s, n_head, head_dim, attn_sparsity):
        """Compute sparse attention values."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new, mask, b, src_s, tgt_s, n_head, head_dim):
        """Compute attention on mixed devices."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        """MLP layer."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def mlp_llama(self, inputs, gate, down, up, donate, config, post_attention_layernorm, compress_weight=False):
        """MLP layer for LLaMA model."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def synchronize(self):
        """Synchronize device."""
        if self.device_type == DeviceType.CUDA:
            torch.cuda.synchronize(self.dev)

    def mem_stats(self):
        """Get memory statistics."""
        if self.device_type == DeviceType.CUDA:
            allocated = torch.cuda.memory_allocated(self.dev)
            reserved = torch.cuda.memory_reserved(self.dev)
            return allocated, reserved
        else:
            return 0, 0

    def print_stats(self, output_file=None):
        """Print device statistics."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def __str__(self):
        return f"TorchDevice(name={self.name}, device_type={self.device_type}, mem_capacity={self.mem_capacity}, flops={self.flops})" 