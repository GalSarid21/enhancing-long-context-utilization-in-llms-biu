import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaForCausalLM, LlamaModel


class PiecewiseRoPE(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, factors=None):
        super().__init__(dim, max_position_embeddings, base, device)
        # Your piecewise logic here
        self.factors = torch.tensor(factors or [1.0], device=device)
        self.seg_len = max_position_embeddings // len(self.factors)

    def forward(self, x, position_ids, seq_len=None):
        # Determine factor based on position
        idx = (position_ids // self.seg_len).clamp(0, len(self.factors) - 1)
        current_factor = self.factors[idx]
        
        # Apply scaling to positions before frequency calculation
        # Note: seq_len / factor logic
        scaled_pos = position_ids.to(x.dtype) / current_factor
        
        # ... standard RoPE math follows ...
        return super().forward(x, scaled_pos, seq_len)


class PiecewiseLlamaModel(LlamaModel):
    def _init_rope(self):
        # Retrieve your custom factors from the config
        factors = getattr(self.config, "rope_piecewise_factors", [1.0])
        
        # Initialize our custom RoPE class
        self.rotary_emb = PiecewiseRoPE(
            self.config.hidden_size // self.config.num_attention_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            factors=factors,
        )


class PiecewiseLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Override the base model with our custom PiecewiseLlamaModel
        self.model = PiecewiseLlamaModel(config)
        self.post_init()
