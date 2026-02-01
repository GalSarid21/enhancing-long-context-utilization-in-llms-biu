import os
import json
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaModel, LlamaForCausalLM


class PiecewiseRoPE(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, factors=None, base_factor=32.0):
        super().__init__(dim, max_position_embeddings, base, device)
        # We store the base factor (e.g., 32.0) and the multipliers (e.g., [0.95, 1, 1.05])
        self.multipliers = torch.tensor(factors or [1.0], device=device)
        self.base_factor = base_factor
        self.num_segments = len(self.multipliers)
        self.segment_size = max_position_embeddings // self.num_segments

    def forward(self, x, position_ids, seq_len=None):
        # Determine the segment index for each position
        segment_indices = (position_ids // self.segment_size).clamp(0, self.num_segments - 1)
        
        # Calculate the final factor: base_factor * multiplier
        # Example: 32.0 * 0.95 = 30.4 for the first segment
        current_multipliers = self.multipliers[segment_indices]
        final_effective_factors = self.base_factor * current_multipliers
        
        # Scale positions: position_ids / effective_factor
        scaled_positions = position_ids.to(x.dtype) / final_effective_factors
        
        return super().forward(x, scaled_positions, seq_len)


class PiecewiseLlamaModel(LlamaModel):
    def _init_rope(self):
        # The config object is passed to this model automatically by vLLM
        factors = getattr(self.config, "rope_piecewise_factors", [1.0])
        
        self.rotary_emb = PiecewiseRoPE(
            self.config.hidden_size // self.config.num_attention_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            factors=factors,
        )


class PiecewiseLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = PiecewiseLlamaModel(config)
        self.post_init()
