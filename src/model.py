from dataclasses import dataclass

import torch
from torch import nn

from src.data import DEFAULT_MAX_MIRNA_LEN, DEFAULT_MAX_TARGET_LEN, PAD_ID, TOKEN_TO_ID


@dataclass
class RNAPairTransformerConfig:
    vocab_size: int = len(TOKEN_TO_ID)
    pad_id: int = PAD_ID
    max_target_len: int = DEFAULT_MAX_TARGET_LEN
    max_mirna_len: int = DEFAULT_MAX_MIRNA_LEN
    d_model: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


def masked_mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden_states.dtype)
    masked_sum = (hidden_states * mask).sum(dim=1)
    valid_token_count = mask.sum(dim=1).clamp_min(1.0)
    return masked_sum / valid_token_count


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        key_value_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        key_padding_mask = ~key_value_mask

        attention_output, attention_weights = self.attention(
            query=query_states,
            key=key_value_states,
            value=key_value_states,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden_states = self.norm1(query_states + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(feed_forward_output))
        return hidden_states, attention_weights


class RNAPairTransformer(nn.Module):
    def __init__(self, config: RNAPairTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or RNAPairTransformerConfig()

        if self.config.d_model % self.config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.token_embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.d_model,
            padding_idx=self.config.pad_id,
        )
        self.target_position_embedding = nn.Embedding(
            self.config.max_target_len,
            self.config.d_model,
        )
        self.mirna_position_embedding = nn.Embedding(
            self.config.max_mirna_len,
            self.config.d_model,
        )
        self.embedding_dropout = nn.Dropout(self.config.dropout)

        self.target_encoder = self._build_encoder()
        self.mirna_encoder = self._build_encoder()
        self.cross_attention = CrossAttentionBlock(
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, 1),
        )

    def _build_encoder(self) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
        )
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config.num_encoder_layers,
        )

    def _embed_tokens(
        self,
        token_ids: torch.Tensor,
        position_embedding: nn.Embedding,
    ) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = position_embedding(positions)
        return self.embedding_dropout(token_embeddings + position_embeddings)

    def forward(
        self,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        mirna_ids: torch.Tensor,
        mirna_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        target_hidden = self._embed_tokens(target_ids, self.target_position_embedding)
        mirna_hidden = self._embed_tokens(mirna_ids, self.mirna_position_embedding)

        target_hidden = self.target_encoder(
            target_hidden,
            src_key_padding_mask=~target_mask,
        )
        mirna_hidden = self.mirna_encoder(
            mirna_hidden,
            src_key_padding_mask=~mirna_mask,
        )

        cross_attended_hidden, attention_weights = self.cross_attention(
            query_states=mirna_hidden,
            key_value_states=target_hidden,
            key_value_mask=target_mask,
            return_attention=return_attention,
        )

        pooled_hidden = masked_mean_pool(cross_attended_hidden, mirna_mask)
        logits = self.classifier(pooled_hidden).squeeze(-1)

        outputs = {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
        }
        if return_attention and attention_weights is not None:
            outputs["attention_weights"] = attention_weights
        return outputs
