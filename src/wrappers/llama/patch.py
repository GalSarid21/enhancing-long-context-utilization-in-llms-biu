import torch
import logging
from typing import Tuple, List, Any

logger = logging.getLogger("vllm")

def apply_piecewise_monkeypatch(
    multipliers: List[float], 
    max_position_embeddings: int = 131072
) -> None:
    """
    Research Implementation: Phase-Continuous Cumulative RoPE.
    
    This 'Stitched' approach allows for non-monotonic scaling factors 
    (e.g., [1.5, 0.95, 2.0]) by calculating cumulative offsets that ensure 
    the 'Virtual Position' (rotation angle) remains continuous across boundaries.
    """

    # 1. THE FORWARD PASS (Continuous Odometer)
    def piecewise_forward_native(
        self: Any, 
        positions: torch.Tensor, 
        query: torch.Tensor, 
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        num_tokens: int = query.shape[0]
        head_size: int = getattr(self, "head_size", 128)
        dtype, device = query.dtype, query.device
        
        flat_positions = positions.flatten()
        s_size = self._segment_size
        
        # Determine segments
        m_tensor = self._piecewise_multipliers.to(device=device)
        o_tensor = self._piecewise_offsets.to(device=device)
        
        segment_indices = (flat_positions // s_size).clamp(0, len(m_tensor) - 1)
        
        # v_pos = Cumulative_Offset + (Distance_into_current_segment / Current_Multiplier)
        segment_starts = segment_indices * s_size
        rel_positions = flat_positions - segment_starts
        
        current_m = m_tensor[segment_indices].to(dtype=torch.float32)
        current_o = o_tensor[segment_indices].to(dtype=torch.float32)
        
        # The 'Odometer' math
        v_pos = current_o + (rel_positions.float() / current_m)
        
        # Lookup from native Su-scaled cache
        lookup_indices = v_pos.round().long().clamp(0, self.cos_sin_cache.shape[0] - 1)
        cos_sin = self.cos_sin_cache.index_select(0, lookup_indices)
        cos, sin = cos_sin.chunk(2, dim=-1)

        # Broadcast and Rotate
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        query, key = query.view(num_tokens, -1, head_size), key.view(num_tokens, -1, head_size)
        return (query * cos + rotate_half(query) * sin).flatten(1), \
               (key * cos + rotate_half(key) * sin).flatten(1)

    # 2. INITIALIZATION (The Stitching Logic)
    import vllm.model_executor.layers.rotary_embedding as vllm_rope
    original_init = vllm_rope.Llama3RotaryEmbedding.__init__

    def patched_rope_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        
        if not hasattr(vllm_rope.Llama3RotaryEmbedding, "_initialized_stitch"):
            s_size = max_position_embeddings // len(multipliers)
            
            # THE STITCH: Calculate how far the 'Virtual Position' has traveled 
            # by the end of each segment to set the starting point for the next.
            offsets = [0.0]
            current_virtual_mileage = 0.0
            for i in range(len(multipliers) - 1):
                # Mileage added = Actual tokens in segment / Multiplier of that segment
                current_virtual_mileage += s_size / multipliers[i]
                offsets.append(current_virtual_mileage)
            
            vllm_rope.Llama3RotaryEmbedding._piecewise_multipliers = torch.tensor(multipliers)
            vllm_rope.Llama3RotaryEmbedding._piecewise_offsets = torch.tensor(offsets)
            vllm_rope.Llama3RotaryEmbedding._segment_size = s_size
            vllm_rope.Llama3RotaryEmbedding._initialized_stitch = True
            
            logger.info(f"STITCHED PIECEWISE ROPE ACTIVE")
            logger.info(f"Segments: {len(multipliers)} | Multipliers: {multipliers}")
            logger.info(f"Calculated Continuous Offsets: {offsets}")

    vllm_rope.Llama3RotaryEmbedding.__init__ = patched_rope_init
    vllm_rope.Llama3RotaryEmbedding.forward = piecewise_forward_native
