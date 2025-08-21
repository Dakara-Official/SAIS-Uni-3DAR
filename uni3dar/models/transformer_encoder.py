from functools import partial
import torch
from torch import nn
import torch.utils
from unicore.modules import RMSNorm

from .layers import DropPath, FeedForward
from .attention import Attention


class TransformerLayer(nn.Module):

    def __init__(
        self,
        dim,
        heads,
        mlp_dim,
        residual_dropout=0.1,
        attn_dropout=0.1,
        rope=None,
        deterministic=False,
        causal=False,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=heads,
            dropout=attn_dropout,
            bias=False,
            rope=rope,
            deterministic=deterministic,
            causal=causal,
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(
            dim,
            mlp_dim,
            256,
        )

        self.dropout = DropPath(residual_dropout)

    def forward(
        self,
        x,
        cu_lens=None,
        max_len=None,
        kv_cache=None,
    ):
        x = x + self.dropout(
            self.attn(self.attn_norm(x), cu_lens, max_len, kv_cache=kv_cache)
        )
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        residual_dropout=0.1,
        attn_dropout=0.1,
        rope=None,
        checkpoint_activation_threshold=140000,
        deterministic=False,
        causal=False,
    ):
        super().__init__()
        self.checkpoint_activation_threshold = checkpoint_activation_threshold
        self.layers = nn.ModuleList([])
        droppath_probs = [x.item() for x in torch.linspace(0, residual_dropout, depth)]
        for i in range(depth):
            self.layers.append(
                TransformerLayer(
                    dim,
                    heads,
                    mlp_dim,
                    droppath_probs[i],
                    attn_dropout,
                    rope=rope,
                    deterministic=deterministic,
                    causal=causal,
                )
            )

    def forward(
        self,
        x,
        cu_lens=None,
        max_len=None,
        kv_cache=None,
    ):

        layers = [
            partial(
                b,
                cu_lens=cu_lens,
                max_len=max_len,
            )
            for b in self.layers
        ]
        if (
            cu_lens is not None
            and cu_lens[-1] > self.checkpoint_activation_threshold
            and self.training
        ):
            for i, layer in enumerate(layers):
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for i, layer in enumerate(self.layers):
                cur_kv_cache = None if kv_cache is None else kv_cache[i]
                x = layer(
                    x,
                    cu_lens,
                    max_len,
                    cur_kv_cache,
                )
                if kv_cache is not None:
                    kv_cache[i] = cur_kv_cache
        return x
