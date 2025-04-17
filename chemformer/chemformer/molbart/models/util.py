import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention block
        att = self.norm1(src)
        att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        att = src + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return out


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Self attention block
        query = self.norm1(tgt)
        query = self.self_attn(
            query,
            query,
            query,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        query = tgt + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(
            att,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return out


def imq_kernel(z1, z2):
    batch_size = z1.size(0)
    dists_z1 = ((z1[:, None] - z1[None]) ** 2).sum(-1)
    dists_z2 = ((z2[:, None] - z2[None]) ** 2).sum(-1)
    dists_z1_z2 = ((z1[:, None] - z2[None]) ** 2).sum(-1)
    I_matrix = torch.eye(batch_size).to(z1)
    stats = 0.0
    z_dim = z1.size(-1)
    for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        C = 2 * z_dim * 1.0 * scale
        res1 = C / (C + dists_z1) + C / (C + dists_z2)
        res1 = (1.0 - I_matrix) * res1
        res1 = res1.sum() / (batch_size - 1)

        res2 = C / (C + dists_z1_z2)
        res2 = res2.sum() * 2.0 / (batch_size)
        stats += res1 - res2
    return stats / batch_size