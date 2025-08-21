# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
from .layers import Linear
from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func
from functools import lru_cache


@lru_cache(maxsize=16)
def get_causal_mask(seq_q, seq_k, device):
    offset = seq_k - seq_q
    i = torch.arange(seq_q, device=device).unsqueeze(1)
    j = torch.arange(seq_k, device=device).unsqueeze(0)
    causal_mask = (j > (offset + i)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return causal_mask


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=False,
        rope=None,
        deterministic=False,
        causal=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, init="bert")
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, init="bert")
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, init="bert")
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, init="final")
        self.rope = rope
        self.scale = self.head_dim**-0.5
        self.deterministic = deterministic
        self.causal = causal

    def forward(
        self,
        query,
        cu_lens,
        max_len,
        kv_cache=None,
    ) -> Tensor:

        query_size = query.size()
        assert query_size[-1] == self.embed_dim
        q, k, v = (
            self.q_proj(query).view(*query_size[:-1], self.num_heads, -1),
            self.k_proj(query).view(*query_size[:-1], self.num_heads, -1),
            self.v_proj(query).view(*query_size[:-1], self.num_heads, -1),
        )
        if self.rope is not None:
            q, k = self.rope.apply_qk(q, k)
        if kv_cache is None:
            cu_lens_window = cu_lens
            assert cu_lens_window is not None
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_lens_window,
                cu_lens_window,
                max_len,
                max_len,
                (self.dropout if self.training else 0.0),
                deterministic=self.deterministic,
                causal=self.causal,
            )
        else:
            assert not self.training
            assert len(q.shape) == 4, (q.shape, query_size)

            # q: [B, 1, H, D], k: [B, H, 1, D], v: [B, H, 1, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # k: [B, H, 1, D] -> [B, H, L, D]
            if "k" in kv_cache:
                k_cache, v_cache = kv_cache["k"], kv_cache["v"]
                k = torch.cat([k_cache, k], dim=2)
                del k_cache, kv_cache["k"]
                v = torch.cat([v_cache, v], dim=2)
                del v_cache, kv_cache["v"]

            seq_q = q.shape[1]
            if seq_q >= 128:
                # use flash_attn to reduce peak memory usage
                out = flash_attn_func(
                    q,
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    dropout_p=0.0,
                    deterministic=self.deterministic,
                    causal=self.causal,
                )
            else:
                q = q * self.scale
                # q: [B, H, 1, D], k: [B, H, D, L] -> attn [B, H, 1, L]
                attn = q.permute(0, 2, 1, 3) @ k.transpose(-1, -2)
                seq_k = attn.shape[-1]
                if self.causal and seq_q > 1:
                    mask = get_causal_mask(seq_q, seq_k, attn.device)
                    attn.masked_fill_(mask, float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                # attn: [B, H, 1, L], v: [B, L, H, D]
                # [B, H, 1, L] @ [B, H, L, D] -> [B, H, 1, D]
                out = attn @ v
                # [B, H, 1, D] -> [B, 1, H, D]
                out = out.permute(0, 2, 1, 3).contiguous()

            cur_step = kv_cache["step"]
            kv_cache["k"] = k
            kv_cache["v"] = v
            kv_cache["step"] = cur_step + 1

        out = out.view(*query_size[:-1], -1)
        return self.out_proj(out)
