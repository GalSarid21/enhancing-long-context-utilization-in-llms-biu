import torch
import logging
from typing import Tuple, List, Any

logger = logging.getLogger("vllm")

def apply_piecewise_monkeypatch(
    multipliers: List[float], 
    max_position_embeddings: int = 131072
) -> None:
    """
    Research Implementation: Multi-Cache Piecewise Rotary Positional Embedding (RoPE).

    1. THE PIECEWISE CONCEPT:
    This patch implements the hypothesis that different regions of a long-context 
    window benefit from varying positional 'densities'. By dividing the context 
    into discrete segments, we apply specific scaling factors to each region.

    2. MULTI-CACHE APPROACH:
    Llama 3.2 uses non-linear 'Su-scaling'. To preserve architectural integrity, 
    this implementation dynamically instantiates N unique Llama3RotaryEmbedding 
    caches. During the forward pass, tokens are masked based on position and 
    mapped to their corresponding frequency table.

    3. RESEARCH CONTEXT:
    Non-standard monkeypatching for experimental evaluation of context 
    utilization in decoder-only transformers within vLLM 0.8.4.
    """

    # 1. THE FORWARD PASS (Switching logic)
    def piecewise_forward_native(
        self: Any, 
        positions: torch.Tensor, 
        query: torch.Tensor, 
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        num_tokens: int = query.shape[0]
        head_size: int = getattr(self, "head_size", 128)
        dtype = query.dtype
        device = query.device
        
        flat_positions: torch.Tensor = positions.flatten()
        s_size: int = self._segment_size
        
        # Determine segment index for each token
        segment_indices: torch.Tensor = (flat_positions // s_size).clamp(0, len(self._piecewise_caches) - 1)
        
        cos = torch.empty((num_tokens, head_size // 2), device=device, dtype=dtype)
        sin = torch.empty((num_tokens, head_size // 2), device=device, dtype=dtype)
        
        for i, cache in enumerate(self._piecewise_caches):
            mask = (segment_indices == i)
            if not mask.any():
                continue
            
            seg_positions = flat_positions[mask]
            
            # Ensure cache is on the same device as query/key
            cache_gpu = cache.to(device=device, non_blocking=True)
            cos_sin = cache_gpu.index_select(0, seg_positions.clamp(0, cache.shape[0]-1))
            
            c, s = cos_sin.chunk(2, dim=-1)
            cos[mask] = c.to(dtype=dtype)
            sin[mask] = s.to(dtype=dtype)

        # Broadcast for Llama 3 [tokens, 1, head_size]
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        query = query.view(num_tokens, -1, head_size)
        key = key.view(num_tokens, -1, head_size)
        
        q_out = (query * cos) + (rotate_half(query) * sin)
        k_out = (key * cos) + (rotate_half(key) * sin)
        
        return q_out.flatten(1), k_out.flatten(1)

    # 2. THE INITIALIZATION (Dynamic Cache Building)
    import vllm.model_executor.layers.rotary_embedding as vllm_rope
    original_init = vllm_rope.Llama3RotaryEmbedding.__init__

    def patched_rope_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        
        # Detect native scaling parameters
        native_factor = getattr(self, "scaling_factor", 32.0)
        low_f = getattr(self, "low_freq_factor", 1.0)
        high_f = getattr(self, "high_freq_factor", 4.0)
        orig_max = getattr(self, "original_max_position_embeddings", 8192)
        
        self._piecewise_caches = []
        for i, m in enumerate(multipliers):
            target_factor = m * native_factor
            logger.info(f"Initializing Piecewise Cache {i} with multiplier {m} (Total Factor: {target_factor})")

            temp_rope = vllm_rope.Llama3RotaryEmbedding(
                head_size=self.head_size,
                rotary_dim=self.rotary_dim,
                max_position_embeddings=max_position_embeddings,
                base=self.base,
                is_neox_style=True,
                scaling_factor=target_factor,
                low_freq_factor=low_f,
                high_freq_factor=high_f,
                original_max_position_embeddings=orig_max,
                dtype=self.cos_sin_cache.dtype
            )
            self._piecewise_caches.append(temp_rope.cos_sin_cache)
            
        self._segment_size = max_position_embeddings // len(multipliers)
        
        if not hasattr(vllm_rope.Llama3RotaryEmbedding, "_logged_v2"):
            logger.info(f"Research Patch: Multi-Cache logic fully initialized.")
            vllm_rope.Llama3RotaryEmbedding._logged_v2 = True

    # 3. APPLY OVERWRITE
    vllm_rope.Llama3RotaryEmbedding.__init__ = patched_rope_init
    vllm_rope.Llama3RotaryEmbedding.forward = piecewise_forward_native