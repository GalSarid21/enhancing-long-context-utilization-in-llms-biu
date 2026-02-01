import torch
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaModel, LlamaForCausalLM


class PiecewiseWarpedRoPE(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=131072, base=500000, device=None, factors=None, rope_scaling=None):
        super().__init__(dim, max_position_embeddings, base, device)
        
        # 1. Capture Meta's Llama 3.1/3.2 parameters
        self.scaling_factor = rope_scaling.get("factor", 32.0)
        self.low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
        self.high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
        self.old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)
        
        # 2. Capture your custom multipliers
        self.multipliers = torch.tensor(factors or [1.0], device=device)
        self.num_segments = len(self.multipliers)
        self.segment_size = max_position_embeddings // self.num_segments

        # 3. Pre-compute the Meta "Warped" inv_freq (The standard Llama 3 logic)
        # This part ensures we keep the high/low freq preservation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        
        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        # Standard Llama 3 frequency warping
        new_inv_freq = torch.where(wavelen > low_freq_wavelen, inv_freq / self.scaling_factor, inv_freq)
        
        smooth_condition = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        smooth_factor = (self.old_context_len / wavelen - self.low_freq_factor) / (self.high_freq_factor - self.low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / self.scaling_factor) + smooth_factor * inv_freq
        
        self.register_buffer("inv_freq", torch.where(smooth_condition, smoothed_inv_freq, new_inv_freq))

    def forward(self, x, position_ids, seq_len=None):
        # 4. Apply your Piecewise Multipliers at runtime
        segment_indices = (position_ids // self.segment_size).clamp(0, self.num_segments - 1)
        current_multipliers = self.multipliers[segment_indices]
        
        # We multiply the position IDs by 1/multiplier to shift the scale
        # Effective factor = Base Scaling * Multiplier
        scaled_positions = position_ids.to(x.dtype) / current_multipliers
        
        # Standard RoPE logic using our pre-warped inv_freq
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = scaled_positions[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class PiecewiseLlamaModel(LlamaModel):
    def _init_rope(self):
        factors = getattr(self.config, "rope_piecewise_factors", [1.0])
        # Use the base factor from the original config (usually 32.0)
        base_factor = getattr(self.config.rope_scaling, "factor", 32.0)
        
        self.rotary_emb = PiecewiseRoPE(
            self.config.hidden_size // self.config.num_attention_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            factors=factors,
            base_factor=base_factor
        )


class PiecewiseLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = PiecewiseLlamaModel(config)
        self.post_init()
