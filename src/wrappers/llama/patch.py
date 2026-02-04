import torch
import logging
from typing import Tuple, List, Any

logger = logging.getLogger("vllm")

def apply_piecewise_monkeypatch(
    multipliers: List[float], 
    max_position_embeddings: int = 131072
) -> None:
    """
    Implements Piecewise RoPE scaling for Llama 3.2.
    Uses 'Virtual Position' mapping to achieve dynamic scaling factors 
    relative to the native 32x base.
    """

    # 1. THE FORWARD PASS (The 'Stitching' Logic)
    def piecewise_forward_native(
        self: Any, 
        positions: torch.Tensor, 
        query: torch.Tensor, 
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        num_tokens: int = query.shape[0]
        head_size: int = getattr(self, "head_size", 128)
        
        # Instance attributes from patched __init__
        m_tensor = self._piecewise_multipliers.to(device=positions.device)
        s_size: int = self._segment_size
        
        flat_positions: torch.Tensor = positions.flatten()
        
        # Select the 'best' multiplier for this token's range
        segment_indices: torch.Tensor = (flat_positions // s_size).clamp(0, len(m_tensor) - 1)
        
        # Fetch the pre-multiplied combined factor (e.g. 1.5 * 32)
        # We use float32 for precise division at high token counts
        current_combined_factors: torch.Tensor = m_tensor[segment_indices].to(
            device=positions.device, dtype=torch.float32
        )
        
        # Virtual Position: Actual / (Experimental_Multiplier * 32.0)
        scaled_positions: torch.Tensor = (flat_positions.float() / current_combined_factors).round().long()
        
        # Clamp to avoid indexing outside the 128k (or extended) cache
        lookup_indices: torch.Tensor = scaled_positions.clamp(0, self.cos_sin_cache.shape[0] - 1)

        # Lookup from the native Su-scaled Llama 3.2 cache
        cos_sin: torch.Tensor = self.cos_sin_cache.index_select(0, lookup_indices)
        cos, sin = cos_sin.chunk(2, dim=-1)

        if cos.shape[-1] != head_size:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        query = query.view(num_tokens, -1, head_size)
        key = key.view(num_tokens, -1, head_size)
        
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        q_out: torch.Tensor = (query * cos) + (rotate_half(query) * sin)
        k_out: torch.Tensor = (key * cos) + (rotate_half(key) * sin)
        
        return q_out.flatten(1), k_out.flatten(1)

    # 2. THE INITIALIZATION (Factor Pre-calculation)
    import vllm.model_executor.layers.rotary_embedding as vllm_rope
    original_init = vllm_rope.Llama3RotaryEmbedding.__init__

    def patched_rope_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        
        native_factor: float = getattr(self, "scaling_factor", 32.0)
        
        # Pre-align your multipliers to the model's expected 32x baseline
        combined_multipliers: List[float] = [m * native_factor for m in multipliers]
        
        self._piecewise_multipliers = torch.tensor(
            combined_multipliers, 
            dtype=torch.float32 
        )
        self._segment_size = max_position_embeddings // len(multipliers)
        
        if not hasattr(vllm_rope.Llama3RotaryEmbedding, "_logged_once"):
            logger.info(f"THESIS MODE: Piecewise RoPE Scaling Enabled")
            logger.info(f"Baseline Factor: {native_factor}")
            logger.info(f"Combined Factors: {combined_multipliers}")
            vllm_rope.Llama3RotaryEmbedding._logged_once = True

    # 3. APPLY PATCH
    vllm_rope.Llama3RotaryEmbedding.__init__ = patched_rope_init
    vllm_rope.Llama3RotaryEmbedding.forward = piecewise_forward_native