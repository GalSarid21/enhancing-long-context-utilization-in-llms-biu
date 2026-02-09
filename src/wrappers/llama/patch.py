import torch
import logging
import math
from typing import Tuple, List, Any, Dict

logger = logging.getLogger("vllm")

def apply_piecewise_monkeypatch(
    abs_factors: List[float], 
    max_position_embeddings: int = 131072
) -> None:
    """
    Methodology: Phase-Continuous Piecewise RoPE Scaling.
    
    This implementation introduces a segmented approach to positional encoding, 
    allowing for non-uniform resolution (scaling factors) across the context window.
    
    Key Innovations:
    1. Segmented Scaling: Applies distinct absolute scaling factors to predefined 
       document regions to optimize for variable information density.
    2. Phase-Continuous Integration: Utilizes a cumulative frequency integral 
       (inspired by the MS-PoE methodology) to ensure that the rotational phase 
       remains a continuous function, eliminating 'positional whiplash' at 
       segment boundaries.
    3. Wavelength-Aware Resolution: Integrates Llama 3.2's Su-Scaling zones to 
       preserve high-frequency dimensions, ensuring local syntax remains sharp 
       even in heavily scaled global regions.
    """

    def generate_piecewise_cache(
        self: Any, 
        factors: List[float], 
        max_pos: int, 
        config: Dict[str, float]
    ) -> torch.Tensor:
        head_dim: int = self.head_size
        device: torch.device = self.cos_sin_cache.device
        dtype: torch.dtype = self.cos_sin_cache.dtype
        
        # Base Llama 3 frequencies
        inv_freq: torch.Tensor = 1.0 / (float(self.base) ** (
            torch.arange(0, head_dim, 2).float().to(device) / head_dim
        ))
        
        # Su-Scaling Thresholds
        low_w: float = config['o_max'] / config['low_f']
        high_w: float = config['o_max'] / config['high_f']

        # Constructing the Piecewise Frequency Map
        s_size: int = max_pos // len(factors)
        full_freq_map: torch.Tensor = torch.zeros((max_pos, head_dim // 2), device=device)
        
        for idx, f_val in enumerate(factors):
            start: int = idx * s_size
            end: int = (idx + 1) * s_size if idx < len(factors) - 1 else max_pos
            
            # Compute Su-scaled frequencies for the current segment's factor
            segment_freqs: List[float] = []
            for f in inv_freq.tolist():
                w: float = 2 * math.pi / f
                if w < high_w:
                    segment_freqs.append(f) # Zone 1: No scaling (local precision)
                elif w > low_w:
                    segment_freqs.append(f / f_val) # Zone 3: Absolute Piecewise Scaling
                else:
                    # Zone 2: Smooth interpolation ramp
                    smooth: float = (config['o_max'] / w - config['low_f']) / (config['high_f'] - config['low_f'])
                    segment_freqs.append((1 - smooth) * (f / f_val) + smooth * f)
            
            full_freq_map[start:end] = torch.tensor(segment_freqs, device=device)

        # MS-PoE inspired Phase Integration
        # We accumulate frequencies to maintain a continuous rotational timeline
        phases: torch.Tensor = torch.cumsum(full_freq_map, dim=0)
        # Shift to start at zero phase
        phases = torch.roll(phases, shifts=1, dims=0)
        phases[0] = 0.0

        return torch.cat([phases.cos(), phases.sin()], dim=-1).to(dtype)

    def forward_piecewise(
        self: Any, 
        positions: torch.Tensor, 
        query: torch.Tensor, 
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        num_tokens: int = query.shape[0]
        h_size: int = getattr(self, "head_size", 128)
        
        # High-resolution phase lookup
        cos_sin: torch.Tensor = self.cos_sin_cache.index_select(0, positions.flatten())
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        if cos.shape[-1] != h_size:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        query = query.view(num_tokens, -1, h_size)
        key = key.view(num_tokens, -1, h_size)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        
        return (query * cos + rotate_half(query) * sin).flatten(1), \
               (key * cos + rotate_half(key) * sin).flatten(1)

    # Injection Logic
    import vllm.model_executor.layers.rotary_embedding as vllm_rope
    original_init = vllm_rope.Llama3RotaryEmbedding.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if not hasattr(vllm_rope.Llama3RotaryEmbedding, "_piecewise_global_cache"):
            config: Dict[str, float] = {
                'low_f': float(getattr(self, "low_freq_factor", 1.0)),
                'high_f': float(getattr(self, "high_freq_factor", 4.0)),
                'o_max': float(getattr(self, "old_context_len", 8192))
            }
            logger.info("--- PIECEWISE PHASE-CONTINUOUS SCALING ACTIVE ---")
            logger.info(f"Targets: {abs_factors}")
            
            cache: torch.Tensor = generate_piecewise_cache(
                self, abs_factors, max_position_embeddings + 512, config
            )
            vllm_rope.Llama3RotaryEmbedding._piecewise_global_cache = cache
        
        self.cos_sin_cache = vllm_rope.Llama3RotaryEmbedding._piecewise_global_cache

    vllm_rope.Llama3RotaryEmbedding.__init__ = patched_init
    vllm_rope.Llama3RotaryEmbedding.forward = forward_piecewise