import torch
import logging
from typing import Tuple, List, Any
import vllm.model_executor.layers.rotary_embedding as vllm_rope
from vllm.worker.worker import Worker

logger = logging.getLogger("vllm")

def apply_piecewise_monkeypatch(
    multipliers: List[float], 
    max_position_embeddings: int = 131072
) -> None:
    """
    Type-hinted monkeypatch for piecewise RoPE scaling in vLLM 0.8.4.
    """
    
    num_segments: int = len(multipliers)
    segment_size: int = max_position_embeddings // num_segments

    # 1. THE INJECTION LOGIC
    def piecewise_forward_native(
        self: Any, 
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        flat_positions: torch.Tensor = positions.flatten()
        
        # DEFENSIVE ATTRIBUTE RETRIEVAL
        # We provide defaults so it doesn't crash during vLLM's initial profiling
        m_tensor: torch.Tensor = getattr(self, "_piecewise_multipliers", None)
        n_segs: int = getattr(self, "_num_segments", num_segments)
        s_size: int = getattr(self, "_segment_size", segment_size)

        # Fallback: if profiling happens before Worker.__init__ finishes, use standard scaling
        if m_tensor is None:
            return original_rope_forward(self, positions, query, key)

        # PIECEWISE LOGIC
        segment_indices: torch.Tensor = (flat_positions // s_size).clamp(0, n_segs - 1)
        current_multipliers: torch.Tensor = m_tensor[segment_indices].unsqueeze(-1)
        
        # INTERPOLATION MATH
        # Cast to float for division, then back to long for indexing
        scaled_positions: torch.Tensor = (flat_positions.float() / current_multipliers.squeeze(-1)).long()

        # Index the cos/sin cache with scaled positions
        cos_sin: torch.Tensor = self.cos_sin_cache.index_select(0, scaled_positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Apply rotation (query and key are mutated or returned depending on vLLM version)
        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        return query, key

    # 2. PERFORM THE INJECTION
    # Store the original forward so we can fall back to it if needed
    original_rope_forward = vllm_rope.RotaryEmbedding.forward
    vllm_rope.RotaryEmbedding.forward = piecewise_forward_native

    # 3. WORKER-LEVEL ATTRIBUTE DEFINITION
    original_worker_init = Worker.__init__

    def patched_worker_init(self: Worker, *args: Any, **kwargs: Any) -> None:
        # ATTACH TO CLASS: This ensures 'self' in the forward pass can see them
        if not hasattr(vllm_rope.RotaryEmbedding, "_piecewise_multipliers"):
            setattr(vllm_rope.RotaryEmbedding, "_piecewise_multipliers", torch.tensor(
                multipliers, 
                device="cuda", 
                dtype=torch.bfloat16 
            ))
            setattr(vllm_rope.RotaryEmbedding, "_num_segments", num_segments)
            setattr(vllm_rope.RotaryEmbedding, "_segment_size", segment_size)
            
            logger.info(f"[POC ACTIVE] Piecewise attributes defined on RotaryEmbedding class.")
        
        original_worker_init(self, *args, **kwargs)

    Worker.__init__ = patched_worker_init
    logger.info("vLLM Worker initialization patched.")