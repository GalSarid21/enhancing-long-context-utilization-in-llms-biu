import torch
import logging
import math
from typing import List, Any

logger = logging.getLogger("vllm")

def apply_piecewise_monkeypatch(
    abs_factors: List[float], 
    max_position_embeddings: int = 131072
) -> None:
    """
    Research Implementation: Frequency-Aware Absolute Stitched RoPE.
    
    1. Locks High-Freq dimensions at 1.0 (Protects 'Cyrus' retrieval).
    2. Applies Absolute Factors (48.0, 30.4, 64.0) to Low-Freq dimensions.
    3. Uses 'Odometer' phase calculation for perfect continuity.
    """

    def calculate_frequency_matrix(head_size, max_pos, base, f_list, s_size, config):
        # 1. Base frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
        
        # 2. Thresholds for Llama 3.2 Su-scaling
        low_w = config['o_max'] / config['low_f']   # ~8192
        high_w = config['o_max'] / config['high_f'] # ~2048
        
        full_phases = torch.zeros((max_pos, head_size // 2))
        
        # 3. Calculate 'Speed Sets' for each Factor
        segment_freq_sets = []
        for target_f in f_list:
            freqs = []
            for f in inv_freq:
                wavelen = 2 * math.pi / f
                if wavelen < high_w:
                    freqs.append(f) # PROTECT HIGH FREQ: Stay at 1.0 scale
                elif wavelen > low_w:
                    freqs.append(f / target_f) # SCALE LOW FREQ: Use your factors
                else:
                    # SMOOTH INTERPOLATION (The Su-scaling ramp)
                    smooth = (config['o_max'] / wavelen - config['low_f']) / (config['high_f'] - config['low_f'])
                    freqs.append((1 - smooth) * (f / target_f) + smooth * f)
            segment_freq_sets.append(torch.tensor(freqs))

        # 4. THE ODOMETER (Cumulative Integral)
        # We accumulate the angles token-by-token to ensure zero 'jumps'
        current_phase = torch.zeros(head_size // 2)
        for p in range(max_pos):
            seg_idx = min(p // s_size, len(f_list) - 1)
            full_phases[p] = current_phase
            current_phase += segment_freq_sets[seg_idx]
            
        return torch.cat((full_phases.cos(), full_phases.sin()), dim=-1)

    # Standard Forward (Now using our Frequency-Aware cache)
    def piecewise_forward_native(self, positions, query, key):
        cos_sin = self.cos_sin_cache.to(device=query.device).index_select(0, positions.flatten())
        cos, sin = cos_sin.chunk(2, dim=-1)
        if cos.shape[-1] != getattr(self, "head_size", 128):
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        query, key = query.view(query.shape[0], -1, cos.shape[-1]), key.view(key.shape[0], -1, cos.shape[-1])
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return (query * cos + rotate_half(query) * sin).flatten(1), \
               (key * cos + rotate_half(key) * sin).flatten(1)

    import vllm.model_executor.layers.rotary_embedding as vllm_rope
    original_init = vllm_rope.Llama3RotaryEmbedding.__init__

    def patched_rope_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        
        if not hasattr(vllm_rope.Llama3RotaryEmbedding, "_global_final_cache"):
            config = {
                'low_f': getattr(self, "low_freq_factor", 1.0),
                'high_f': getattr(self, "high_freq_factor", 4.0),
                'o_max': float(getattr(self, "old_context_len", 8192))
            }
            s_size = max_position_embeddings // len(abs_factors)
            
            logger.info(f"STITCHING FREQUENCY-AWARE CACHE: {abs_factors}")
            cache = calculate_frequency_matrix(
                self.head_size, max_position_embeddings + 1024, 
                float(self.base), abs_factors, s_size, config
            )
            vllm_rope.Llama3RotaryEmbedding._global_final_cache = cache.to(dtype=self.cos_sin_cache.dtype)

        self.cos_sin_cache = vllm_rope.Llama3RotaryEmbedding._global_final_cache

    vllm_rope.Llama3RotaryEmbedding.__init__ = patched_rope_init
    vllm_rope.Llama3RotaryEmbedding.forward = piecewise_forward_native