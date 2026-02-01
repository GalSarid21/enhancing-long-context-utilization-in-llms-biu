import os
import json
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaModel, LlamaForCausalLM


class PiecewiseRoPE(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, factors=None):
        super().__init__(dim, max_position_embeddings, base, device)
        # 1. Read from Env Var if available, otherwise use factors passed from config
        env_factors = os.getenv("ROPE_PIECEWISE_FACTORS")
        if env_factors:
            self.factors = torch.tensor(json.loads(env_factors), device=device)
        else:
            self.factors = torch.tensor(factors or [1.0], device=device)
            
        self.num_segments = len(self.factors)
        self.segment_size = max_position_embeddings // self.num_segments

    def forward(self, x, position_ids, seq_len=None):
        segment_indices = (position_ids // self.segment_size).clamp(0, self.num_segments - 1)
        current_factors = self.factors[segment_indices]
        scaled_positions = position_ids.to(x.dtype) / current_factors
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
