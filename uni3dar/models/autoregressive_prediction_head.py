import torch
from torch import nn
from .layers import TaskHead, Embedding


class AutoRegressivePredictionHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(
        self,
        feat_size_list,
        emb_dim,
        emb_dropout=0.1,
        head_dropout=0.0,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        self.num_feat = len(feat_size_list)
        self.emb_layers = nn.ModuleList(
            [Embedding(feat_size_list[i], emb_dim) for i in range(self.num_feat - 1)]
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.pred_heads = nn.ModuleList(
            [
                TaskHead(emb_dim, feat_size_list[i], dropout=head_dropout)
                for i in range(self.num_feat)
            ]
        )

    def forward(self, x, target_list=None):
        if target_list is not None:
            feats = [x]
            for i in range(self.num_feat - 1):
                feats.append(self.dropout(self.emb_layers[i](target_list[i])))

            bsz = x.size(0)

            feat = torch.cat(feats, dim=-1).view(bsz, -1, self.emb_dim)
            y = torch.cumsum(feat, dim=-2)
            pred_logits = [self.pred_heads[i](y[:, i, :]) for i in range(self.num_feat)]
            return pred_logits
        else:
            # only predict for the first feature
            return self.pred_heads[0](x)

    def inference(self, x, sampling_func_list, score_ratio, **kwargs):
        """Inference autogressive prediction, with temperature scaling."""
        bsz = x.size(0)
        feat = x.view(bsz, self.emb_dim)
        bsz = feat.size(0)
        pred_xyz = torch.zeros(
            bsz, self.num_feat, device=feat.device, dtype=torch.int32
        )
        score = torch.zeros(bsz, device=feat.device, dtype=torch.float32)
        for i in range(self.num_feat):
            cur_logits = self.pred_heads[i](feat)
            pred_xyz[:, i], cur_score = sampling_func_list[i](cur_logits)
            score += cur_score * score_ratio[i]
            if i < self.num_feat - 1:
                new_emb = self.emb_layers[i](pred_xyz[:, i])
                feat = feat + new_emb
        return pred_xyz, score
