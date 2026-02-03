import torch
import logging
import math
from typing import Tuple, List, Any
import vllm.model_executor.layers.rotary_embedding as vllm_rope

logger = logging.getLogger("vllm")

def apply_piecewise_monkeypatch(
    multipliers: List[float], 
    max_position_embeddings: int = 131072,
    rope_theta: float = 500000.0
) -> None:
    
    min_mult = min(multipliers)
    required_len = int(max_position_embeddings / min_mult) + 2
    
    logger.info(f"--- Piecewise RoPE Setup ---")
    logger.info(f"Multipliers: {multipliers}")
    logger.info(f"Required Cache Length: {required_len}")

    # 1. THE CUSTOM FORWARD
    def piecewise_forward_native(self, positions, query, key):
        num_tokens = query.shape[0]
        head_size = getattr(self, "head_size", 128)
        
        m_tensor = vllm_rope.RotaryEmbedding._piecewise_multipliers
        s_size = vllm_rope.RotaryEmbedding._segment_size
        
        flat_positions = positions.flatten()
        segment_indices = (flat_positions // s_size).clamp(0, len(m_tensor) - 1)
        current_multipliers = m_tensor[segment_indices].to(device=positions.device)
        
        scaled_positions = (flat_positions.float() / current_multipliers).round().long()
        lookup_indices = scaled_positions.clamp(0, self.cos_sin_cache.shape[0] - 1)

        cos_sin = self.cos_sin_cache.index_select(0, lookup_indices)
        cos, sin = cos_sin.chunk(2, dim=-1)

        if cos.shape[-1] != head_size:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        def rotate_half(x):
            return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

        query = query.view(num_tokens, -1, head_size)
        key = key.view(num_tokens, -1, head_size)
        
        q_out = (query * cos.unsqueeze(1)) + (rotate_half(query) * sin.unsqueeze(1))
        k_out = (key * cos.unsqueeze(1)) + (rotate_half(key) * sin.unsqueeze(1))
        return q_out.flatten(1), k_out.flatten(1)

    # 2. THE HARDENED INITIALIZATION (Manual Frequency Calculation)
    original_init = vllm_rope.RotaryEmbedding.__init__

    def patched_rope_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        
        if not hasattr(vllm_rope.RotaryEmbedding, "_global_extended_cache"):
            logger.info(f"--- Building Extended Piecewise Cache (Size: {required_len}) ---")
            
            # Manually reconstruct Llama 3.2 frequencies to be 100% sure
            # These are the standard Llama 3 parameters
            dim = self.rotary_dim
            # Standard RoPE frequencies: theta^(-2i/d)
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
            
            # Build the cache
            t = torch.arange(required_len, device="cuda", dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
            
            vllm_rope.RotaryEmbedding._global_extended_cache = emb.to(dtype=self.cos_sin_cache.dtype)
            vllm_rope.RotaryEmbedding._piecewise_multipliers = torch.tensor(
                multipliers, device="cuda", dtype=torch.float32
            )
            vllm_rope.RotaryEmbedding._segment_size = max_position_embeddings // len(multipliers)

        self.cos_sin_cache = vllm_rope.RotaryEmbedding._global_extended_cache

    vllm_rope.RotaryEmbedding.__init__ = patched_rope_init
    vllm_rope.RotaryEmbedding.forward = piecewise_forward_native
    logger.info("Piecewise Monkeypatch Injected.")
