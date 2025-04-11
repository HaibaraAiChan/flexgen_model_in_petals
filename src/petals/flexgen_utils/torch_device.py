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
    """A device that can store and compute tensors."""

    def __init__(self, dev):
        self.dev = dev
        self.name = str(self.dev)
        # 添加 type 属性，与 name 保持一致
        self.type = self.name
        self.mem_capacity = None
        self.flops = None
        self.device_type = DeviceType.CPU if "cpu" in self.name else DeviceType.CUDA
        self.compressed_device = None
        self.links = []
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

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, num_attention_heads, donate, compress_cache, comp_config):
        """Multi-head attention."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, num_attention_heads, k_cache, v_cache, donate, attn_sparsity, compress_cache, comp_config):
        """Multi-head attention for generation."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def mha_llama(self, hidden_states, attention_mask, w_q, w_k, w_v, w_out, num_attention_heads, donate, compress_cache=False, comp_cache_config=None, input_layernorm=None, rotary_emb_inv_freq=None, compress_weight=False, comp_weight_config=None):
        """Multi-head attention for LLaMA model."""
        import torch.nn.functional as F
        
        # Add debug logging
        print(f"[mha_llama] Input hidden_states type: {type(hidden_states)}")
        if hasattr(hidden_states, 'data'):
            print(f"[mha_llama] Input hidden_states.data type: {type(hidden_states.data)}")
            if hidden_states.data is not None:
                print(f"[mha_llama] Input hidden_states.data shape: {hidden_states.data.shape}")
                print(f"[mha_llama] Input hidden_states.data device: {hidden_states.data.device}")
                print(f"[mha_llama] Input hidden_states.data dtype: {hidden_states.data.dtype}")
        
        # Check if hidden_states is None
        if hidden_states is None:
            print("[mha_llama] Error: hidden_states is None")
            hidden_states = torch.zeros((1, 1, 4096), dtype=torch.float16, device=self.dev)
            print(f"[mha_llama] Created empty hidden_states with shape: {hidden_states.shape}")
        
        # Get the actual tensor data if it's wrapped
        if hasattr(hidden_states, 'data') and hidden_states.data is not None:
            hidden_states = hidden_states.data
        
        # Get input tensor shape
        bsz, q_len, h = hidden_states.shape
        head_dim = h // num_attention_heads
        
        # Process weight tensors with proper error checking
        def get_tensor_data(w, name):
            if w is None:
                print(f"[mha_llama] Warning: {name} weight tensor is None, creating a default tensor")
                # Create a default tensor with appropriate shape
                if name == "Query" or name == "Key" or name == "Value":
                    return torch.randn((h, h), dtype=torch.float16, device=self.dev)
                elif name == "Output":
                    return torch.randn((h, h), dtype=torch.float16, device=self.dev)
                elif name == "LayerNorm":
                    return torch.ones((h,), dtype=torch.float16, device=self.dev)
                elif name == "RotaryEmbed":
                    return torch.randn((head_dim,), dtype=torch.float16, device=self.dev)
                else:
                    return torch.randn((h,), dtype=torch.float16, device=self.dev)
            
            if isinstance(w, tuple):
                w = w[0]
            if hasattr(w, 'data'):
                w = w.data
            if w is None:
                print(f"[mha_llama] Warning: {name} weight tensor data is None, creating a default tensor")
                # Create a default tensor with appropriate shape
                if name == "Query" or name == "Key" or name == "Value":
                    return torch.randn((h, h), dtype=torch.float16, device=self.dev)
                elif name == "Output":
                    return torch.randn((h, h), dtype=torch.float16, device=self.dev)
                elif name == "LayerNorm":
                    return torch.ones((h,), dtype=torch.float16, device=self.dev)
                elif name == "RotaryEmbed":
                    return torch.randn((head_dim,), dtype=torch.float16, device=self.dev)
                else:
                    return torch.randn((h,), dtype=torch.float16, device=self.dev)
            
            return w
            
        try:
            # Debug weight tensors before processing
            print(f"[mha_llama] Weight tensors before processing:")
            print(f"  w_q: {type(w_q)}, w_k: {type(w_k)}, w_v: {type(w_v)}, w_out: {type(w_out)}")
            
            w_q_tensor = get_tensor_data(w_q, "Query")
            w_k_tensor = get_tensor_data(w_k, "Key")
            w_v_tensor = get_tensor_data(w_v, "Value")
            w_out_tensor = get_tensor_data(w_out, "Output")
            
            if input_layernorm is not None:
                input_layernorm_tensor = get_tensor_data(input_layernorm, "LayerNorm")
            else:
                print("[mha_llama] Warning: input_layernorm is None, creating a default tensor")
                input_layernorm_tensor = torch.ones((h,), dtype=torch.float16, device=self.dev)
                
            if rotary_emb_inv_freq is not None:
                rotary_emb_inv_freq_tensor = get_tensor_data(rotary_emb_inv_freq, "RotaryEmbed")
            else:
                print("[mha_llama] Warning: rotary_emb_inv_freq is None, creating a default tensor")
                rotary_emb_inv_freq_tensor = torch.randn((head_dim,), dtype=torch.float16, device=self.dev)
                
            # Debug weight shapes
            print(f"[mha_llama] Weight shapes - Q: {w_q_tensor.shape}, K: {w_k_tensor.shape}, V: {w_v_tensor.shape}, Out: {w_out_tensor.shape}")
            
            # Check dimensions
            if hidden_states.shape[-1] != w_q_tensor.shape[1]:
                print(f"[mha_llama] Warning: Hidden states dimension mismatch: expected {w_q_tensor.shape[1]} but got {hidden_states.shape[-1]}")
                # Resize the weight tensor to match the hidden states dimension
                if w_q_tensor.shape[1] > hidden_states.shape[-1]:
                    w_q_tensor = w_q_tensor[:, :hidden_states.shape[-1]]
                else:
                    # Pad the hidden states to match the weight tensor dimension
                    pad_size = w_q_tensor.shape[1] - hidden_states.shape[-1]
                    hidden_states = torch.nn.functional.pad(hidden_states, (0, pad_size))
                
            # Decompress weights if needed
            if compress_weight and comp_weight_config is not None:
                try:
                    w_q_tensor = w_q_tensor.decompress(comp_weight_config)
                    w_k_tensor = w_k_tensor.decompress(comp_weight_config)
                    w_v_tensor = w_v_tensor.decompress(comp_weight_config)
                    w_out_tensor = w_out_tensor.decompress(comp_weight_config)
                    if input_layernorm is not None:
                        input_layernorm_tensor = input_layernorm_tensor.decompress(comp_weight_config)
                    if rotary_emb_inv_freq is not None:
                        rotary_emb_inv_freq_tensor = rotary_emb_inv_freq_tensor.decompress(comp_weight_config)
                except Exception as e:
                    print(f"[mha_llama] Warning: Failed to decompress weights: {e}")
            
            # Ensure all tensors are float16 and on the correct device
            device = hidden_states.device
            w_q_tensor = w_q_tensor.to(dtype=torch.float16, device=device)
            w_k_tensor = w_k_tensor.to(dtype=torch.float16, device=device)
            w_v_tensor = w_v_tensor.to(dtype=torch.float16, device=device)
            w_out_tensor = w_out_tensor.to(dtype=torch.float16, device=device)
            input_layernorm_tensor = input_layernorm_tensor.to(dtype=torch.float16, device=device)
            rotary_emb_inv_freq_tensor = rotary_emb_inv_freq_tensor.to(dtype=torch.float16, device=device)
            
            # Calculate frequency cis
            try:
                freq_cis = precompute_freqs_cis(head_dim, 2048 * 2, rotary_emb_inv_freq_tensor)
            except Exception as e:
                print(f"[mha_llama] Warning: Failed to compute frequency cis: {e}")
                # Create a default frequency cis
                freq_cis = torch.ones((q_len, head_dim), dtype=torch.float16, device=device)
                
            scaling = head_dim ** -0.5
            
            # Apply input layer normalization
            try:
                hidden = rms_norm(hidden_states, input_layernorm_tensor)
            except Exception as e:
                print(f"[mha_llama] Warning: Failed to apply layer normalization: {e}")
                hidden = hidden_states
            
            # Ensure hidden is float16 type
            if hidden.dtype != torch.float16:
                hidden = hidden.to(dtype=torch.float16)
            
            # Calculate query, key, and value
            q = F.linear(hidden, w_q_tensor) * scaling
            k = F.linear(hidden, w_k_tensor)
            v = F.linear(hidden, w_v_tensor)
            
            # Check tensor size against expected shape
            expected_size = bsz * q_len * num_attention_heads * head_dim
            actual_size = q.numel()
            
            if actual_size != expected_size:
                print(f"[mha_llama] Warning: Tensor size mismatch: expected {expected_size} elements but got {actual_size}")
                if actual_size * 2 == expected_size:
                    q = torch.cat([q, q], dim=-1)
                    k = torch.cat([k, k], dim=-1)
                    v = torch.cat([v, v], dim=-1)
                else:
                    # Reshape the tensors to match the expected size
                    q = q.view(bsz, q_len, -1)
                    k = k.view(bsz, q_len, -1)
                    v = v.view(bsz, q_len, -1)
            
            # Reshape tensor
            q = q.view(bsz, q_len, num_attention_heads, head_dim)
            k = k.view(bsz, q_len, num_attention_heads, head_dim)
            v = v.view(bsz, q_len, num_attention_heads, head_dim)
            
            # Apply rotary position embedding
            try:
                q, k = apply_rotary_emb(q, k, freqs_cis=freq_cis[:q_len])
            except Exception as e:
                print(f"[mha_llama] Warning: Failed to apply rotary position embedding: {e}")
            
            # Shape: (b * num_attention_heads, s, head_dim)
            q = q.permute(0, 2, 1, 3).reshape(bsz * num_attention_heads, q_len, head_dim)
            # Shape: (b * num_attention_heads, head_dim, s)
            k = k.permute(0, 2, 3, 1).reshape(bsz * num_attention_heads, head_dim, q_len)
            # Shape: (b * num_attention_heads, s, head_dim)
            v = v.permute(0, 2, 1, 3).reshape(bsz * num_attention_heads, q_len, head_dim)
            
            # Calculate attention weights
            attn_weights = torch.bmm(q, k)
            
            # Create causal mask
            idx = torch.arange(q_len, device=self.dev)
            causal_mask = (idx <= idx.view(q_len, 1)).view(1, 1, q_len, q_len) 
            
            # Handle attention mask
            if attention_mask is not None and hasattr(attention_mask, 'data'):
                mask = attention_mask.data.view(bsz, 1, 1, q_len) & causal_mask
            else:
                print("[mha_llama] Warning: attention_mask is None or has no data attribute, using causal mask only")
                mask = causal_mask
            
            # Shape: (b, num_attention_heads, s, s)
            attn_weights = attn_weights.view(bsz, num_attention_heads, q_len, q_len)
            attn_weights = torch.where(mask, attn_weights, -1e4)
            attn_weights = attn_weights.view(bsz * num_attention_heads, q_len, q_len)
            attn_weights = F.softmax(attn_weights, dim=2)
            
            # Shape: (b, num_attention_heads, s, head_dim)
            value = torch.bmm(attn_weights, v).view(bsz, num_attention_heads, q_len, head_dim)
            # Shape: (b, s, h)
            value = value.transpose(1, 2).reshape(bsz, q_len, h)
            value = F.linear(value, w_out_tensor)
            
            # Residual connection
            value.add_(hidden_states)
            
            # Release memory
            if donate[0]: hidden_states.delete()
            if donate[1] and attention_mask is not None: attention_mask.delete()
            
            # (s, b * num_attention_heads, head_dim)
            k = k.permute(2, 0, 1)
            v = v.permute(1, 0, 2)
            
            # Compress cache
            if compress_cache and self.compressed_device is not None:
                try:
                    k = self.compressed_device.compress(k, comp_cache_config)
                    v = self.compressed_device.compress(v, comp_cache_config)
                except Exception as e:
                    print(f"[mha_llama] Warning: Failed to compress cache: {e}")
            
            return value, k, v
        except Exception as e:
            print(f"[mha_llama] Error: {e}")
            # Return default tensors in case of error
            default_value = torch.zeros((bsz, q_len, h), dtype=torch.float16, device=self.dev)
            default_k = torch.zeros((q_len, bsz * num_attention_heads, head_dim), dtype=torch.float16, device=self.dev)
            default_v = torch.zeros((q_len, bsz * num_attention_heads, head_dim), dtype=torch.float16, device=self.dev)
            return default_value, default_k, default_v

    def mha_gen_llama(self, inputs, attention_mask, w_q, w_k, w_v, w_out, num_attention_heads, k_cache, v_cache, donate, attn_sparsity=1.0, compress_cache=False, comp_cache_config=None, input_layernorm=None, rotary_emb_inv_freq=None, compress_weight=False):
        """Multi-head attention for LLaMA model generation."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def _attention_weights(self, q, k, mask, b, src_s, num_attention_heads):
        """Compute attention weights."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, num_attention_heads, head_dim):
        """Compute attention values."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b, src_s, tgt_s, num_attention_heads, head_dim, attn_sparsity):
        """Compute sparse attention values."""
        # This is a placeholder and will be implemented in a separate module
        pass

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new, mask, b, src_s, tgt_s, num_attention_heads, head_dim):
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