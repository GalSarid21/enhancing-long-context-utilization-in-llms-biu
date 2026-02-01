import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaModel, LlamaForCausalLM


class PiecewiseRoPE(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, factors=None, base_factor=32.0):
        super().__init__(dim, max_position_embeddings, base, device)
        self.multipliers = torch.tensor(factors or [1.0], device=device)
        self.base_factor = base_factor
        self.num_segments = len(self.multipliers)
        self.segment_size = max_position_embeddings // self.num_segments

    def forward(self, x, position_ids, seq_len=None):
        segment_indices = (position_ids // self.segment_size).clamp(0, self.num_segments - 1)
        current_multipliers = self.multipliers[segment_indices]
        final_effective_factors = self.base_factor * current_multipliers
        scaled_positions = position_ids.to(x.dtype) / final_effective_factors
        return super().forward(x, scaled_positions, seq_len)


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
