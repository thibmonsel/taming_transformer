"""
Taken from https://github.com/thibmonsel/vqvae/blob/master/vqvae/transformer.py
since the Condtionned transformer is exactly the same as VQ-VAE.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Implementation of multiple head self attention layer.
    """

    def __init__(
        self,
        n_heads: int,
        dim_emb: int,
        max_seq_len: int,
        dropout: float = 0.0,
        flash: Optional[bool] = None,
    ) -> None:
        super().__init__()
        assert dim_emb % n_heads == 0

        # key, query, value projections for all heads, but in a batch
        self.att_weights = nn.Linear(dim_emb, 3 * dim_emb, bias=False)
        self.output_proj = nn.Linear(dim_emb, dim_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.n_heads = n_heads
        self.dim_emb = dim_emb
        if flash is None:
            self.flash = hasattr(F, "scaled_dot_product_attention")
        else:
            self.flash = flash

        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only
            # applied to the left in the input sequence
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                    1, 1, max_seq_len, max_seq_len
                ),
            )
            # to make pyright happy
            self.mask: torch.Tensor

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()[:-1]
        x = x.view(batch_size, seq_len, self.n_heads, self.dim_emb // self.n_heads)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size()[:-2] + (self.dim_emb,))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        q, k, v = self.att_weights(x).split(self.dim_emb, dim=2)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0,
                -torch.inf,  # type: ignore
            )
            att = self.softmax(att)
            att = self.attn_dropout(att)
            y = torch.matmul(att, v)

        y = self.merge_heads(y)
        # output projection
        y = self.resid_dropout(self.output_proj(y))
        return y


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        pointwise_mid_modules: list[nn.Module],
    ) -> None:
        super().__init__()
        self.first_layer = nn.Linear(in_dim, mid_dim, bias=False)
        self.mid = nn.ModuleList(pointwise_mid_modules)
        self.second_layer = nn.Linear(mid_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        for layer in self.mid:
            x = layer(x)
        x = self.second_layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        dim_emb: int,
        max_seq_len: int,
        dropout: float = 0.0,
        flash: Optional[bool] = None,
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            n_heads, dim_emb, max_seq_len, dropout, flash
        )
        self.norm1 = nn.LayerNorm(dim_emb)
        self.mlp = FeedForwardBlock(dim_emb, 4 * dim_emb, dim_emb, [nn.GELU()])
        self.norm2 = nn.LayerNorm(dim_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionnedGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        block_size: int,
        dim_emb: int,
        cond_embedding_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
        flash: Optional[bool] = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, dim_emb)
        self.pos_emb = nn.Embedding(block_size, dim_emb)

        self.n_classes = n_classes
        self.unconditional_idx = n_classes  # Index for the "unconditional" class
        self.cond_class_embedding = nn.Embedding(n_classes + 1, cond_embedding_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(n_heads, dim_emb, block_size, dropout, flash)
                for _ in range(n_layers)
            ]
        )
        self.emb_drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(dim_emb, vocab_size)

    def forward(self, x: torch.Tensor, class_ids=None) -> torch.Tensor:
        assert (
            x.shape[1] <= self.block_size
        ), f"""
            Cannot forward sequence of length {x.shape[0]},
            block size is only {self.block_size}
            """
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        if class_ids is None:
            class_ids = torch.full(
                (x.shape[0],), self.unconditional_idx, device=x.device, dtype=torch.long
            )

        class_embeddings = self.cond_class_embedding(class_ids)  # (B, D)
        class_embeddings_expanded = class_embeddings.unsqueeze(1)  #  (B, 1, D)

        combined_embeddings = (
            token_embeddings + position_embeddings + class_embeddings_expanded
        )

        x = self.emb_drop(combined_embeddings)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)
