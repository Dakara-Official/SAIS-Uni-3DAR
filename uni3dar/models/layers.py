import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.modules import RMSNorm


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "bert":
            self._bert_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def _bert_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)


class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super(Embedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self._normal_init()

        if padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()

    def _normal_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = prob

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"prob={self.drop_prob}"


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False, init="bert")
        self.w2 = Linear(hidden_dim, dim, bias=False, init="final")
        self.w3 = Linear(dim, hidden_dim, bias=False, init="bert")

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = Linear(input_dim, inner_dim, bias=False, init="bert")
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = Linear(inner_dim, num_classes, init="final")

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TaskHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.dense = Linear(embed_dim, embed_dim, bias=False, init="bert")
        self.out = Linear(embed_dim, output_dim, bias=False, init="final")

    def forward(self, x, **kwargs):
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.out(x)
        return x


class FourierFeatureMapping:
    def __init__(
        self,
        input_dims,
        num_freqs,
        include_input=True,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    ):
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        max_freq_log2 = num_freqs - 1
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, steps=num_freqs)
        self.freq_bands = freq_bands.unsqueeze(0).repeat(input_dims, 1)

    def forward(self, x):
        freq_bands = self.freq_bands.to(x.device)
        x_proj = x @ freq_bands
        x_proj = torch.cat([fn(x_proj) for fn in self.periodic_fns], dim=-1)
        if self.include_input:
            x_proj = torch.cat([x, x_proj], dim=-1)
        return x_proj


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AdaNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), Linear(dim, 2 * dim))

    def forward(self, x, c):
        if c is None:
            return self.norm_final(x)
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x
