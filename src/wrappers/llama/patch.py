import torch
import logging
from typing import Tuple, List, Any, Optional
import vllm.model_executor.layers.rotary_embedding as vllm_rope

logger = logging.getLogger("vllm")

def apply_piecewise_monkeypatch(
    multipliers: List[float], 
    max_position_embeddings: Optional[int] = 131072
) -> None:
    """
    Piecewise RoPE Patch for Llama 3.2.
    """

    # 1. THE FORWARD PASS
    def piecewise_forward_native(
        self: Any, 
        positions: torch.Tensor, 
        query: torch.Tensor, 
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        num_tokens: int = query.shape[0]
        head_size: int = getattr(self, "head_size", 128)
        
        # Access attributes defined in __init__
        m_tensor: torch.Tensor = self._piecewise_multipliers
        s_size: int = self._segment_size
        
        # positions is Long/Int, we flatten for index mapping
        flat_positions: torch.Tensor = positions.flatten()
        
        # segment_indices: [tokens] -> [0, 1, 2]
        segment_indices: torch.Tensor = (flat_positions // s_size).clamp(0, len(m_tensor) - 1)
        
        # We fetch the factor (e.g. 1.5 * 32) and move to correct device.
        # We use float32 for the division math to prevent precision loss at 128k.
        current_combined_factors: torch.Tensor = m_tensor[segment_indices].to(
            device=positions.device, dtype=torch.float32
        )
        
        # Virtual Position calculation: Actual / (Multiplier * 32)
        # This stretches the positional encoding space as intended.
        scaled_positions: torch.Tensor = (flat_positions.float() / current_combined_factors).round().long()
        
        # Index into the native Llama 3.2 Su-scaled cache
        lookup_indices: torch.Tensor = scaled_positions.clamp(0, self.cos_sin_cache.shape[0] - 1)

        # Lookup and split into Cos/Sin
        cos_sin: torch.Tensor = self.cos_sin_cache.index_select(0, lookup_indices)
        cos, sin = cos_sin.chunk(2, dim=-1)

        # Handle head dimensions for GQA
        if cos.shape[-1] != head_size:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Reshape to [tokens, heads, head_size] for broadcasting
        query = query.view(num_tokens, -1, head_size)
        key = key.view(num_tokens, -1, head_size)
        
        # Perform the RoPE rotation
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        q_out: torch.Tensor = (query * cos) + (rotate_half(query) * sin)
        k_out: torch.Tensor = (key * cos) + (rotate_half(key) * sin)
        
        return q_out.flatten(1), k_out.flatten(1)

    # 2. THE INITIALIZATION INJECTION
    original_init = vllm_rope.Llama3RotaryEmbedding.__init__

    def patched_rope_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Build the original Su-scaled cache first
        original_init(self, *args, **kwargs)
        
        # Fetch the model's native scaling factor (e.g., 32.0)
        native_factor: float = getattr(self, "scaling_factor", 32.0)
        
        # Pre-multiply your multipliers by the native factor
        combined_multipliers: List[float] = [m * native_factor for m in multipliers]
        
        # Store metadata on the instance
        self._piecewise_multipliers = torch.tensor(
            combined_multipliers, 
            device="cpu", # Moved to GPU in forward pass
            dtype=torch.float32 
        )
        self._segment_size = max_position_embeddings // len(multipliers)
        
        if not hasattr(vllm_rope.Llama3RotaryEmbedding, "_logged_once"):
            logger.info(f"Llama 3.2 Piecewise Active")
            logger.info(f"Native Base Factor: {native_factor}")
            logger.info(f"Computed Multipliers: {combined_multipliers}")
            vllm_rope.Llama3RotaryEmbedding._logged_once = True

    # 3. OVERWRITE CLASS METHODS
    vllm_rope.Llama3RotaryEmbedding.__init__ = patched_rope_init
    vllm_rope.Llama3RotaryEmbedding.forward = piecewise_forward_native
    
    logger.info("Monkeypatch Injected. System ready for long-context run.")
