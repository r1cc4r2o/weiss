import numpy as np
import torch
from torch import nn as tnn
from typing import List, Tuple
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum


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


class PositionalEncoding(torch.nn.Module):
    "Implements the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# old
# class _One2Many(torch.nn.Module):
#     def __init__(
#         self,
#         *,
#         vocabulary_size,
#         d_model=256,
#         num_heads=8,
#         num_layers=6,
#         dim_feedforward=2048,
#         dropout=0.1,
#         layer_norm_eps=1e-6
#     ):
#         super().__init__()
#         _locals = locals()
#         self.h_params = {}
#         for k in _locals:
#             if k == "self" or k.startswith("_"):
#                 continue
#             setattr(self, k, _locals[k])
#             self.h_params[k] = _locals[k]

#         self.token_emb = torch.nn.Embedding(vocabulary_size, d_model)
#         self.pe = PositionalEncoding(d_model, dropout)

#         self.transformer = torch.nn.Transformer(
#             d_model=d_model,
#             nhead=num_heads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             layer_norm_eps=layer_norm_eps,
#             batch_first=True,
#         )
#         self.generator = torch.nn.Linear(d_model, vocabulary_size)

#         self.proj = torch.nn.Sequential(
#             torch.nn.Linear(d_model, d_model//2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(d_model//2, d_model//8),
#             torch.nn.ReLU(),
#             torch.nn.Linear(d_model//8, d_model//16),
#         )

#         layer_norm_kwargs = {"elementwise_affine": False}
#         if int(torch.__version__[0]) >= 2:
#             layer_norm_kwargs["bias"] = False

#         self.interpolate_f = torch.nn.Bilinear(d_model, d_model, d_model, bias=False)

#         self.inv_proj = torch.nn.Sequential(
#             torch.nn.Linear(d_model//16, d_model//8),
#             torch.nn.ReLU(),
#             torch.nn.Linear(d_model//8, d_model//2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(d_model//2, d_model),
#             torch.nn.LayerNorm(d_model, **layer_norm_kwargs),
#         )
#         self.current_z = None

#     def compute_padding_mask(self, mask):
#         """
#         0 means attend / 1 means ignore
#         PyTorch wants a tensor of size B*H x N x N
#         where B is the batch size, H is the number of heads, N is
#         the sequence length
#         """
#         pmask = mask.unsqueeze(1)
#         pmask = pmask.expand(len(pmask), pmask.shape[-1], pmask.shape[-1])
#         pmask = pmask.unsqueeze(1)
#         pmask = pmask.expand(
#             len(pmask), self.num_heads, pmask.shape[-1], pmask.shape[-1]
#         )
#         pmask = pmask.reshape((-1, pmask.shape[-1], pmask.shape[-1]))
#         return pmask.to(mask.device)

#     def compute_causal_mask(self, mask):
#         attn_shape = (1, mask.shape[-1], mask.shape[-1])
#         cmask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
#         return cmask.to(mask.device)

#     def save(self, path):
#         state = {"h_params": self.h_params, "weights": self.state_dict()}
#         torch.save(state, path)

#     def nll(self, src, trg, z=None):
#         with torch.no_grad():
#             self.eval()
#             # remove all the tokens
#             # after the first eos
#             # pad: 0, bos: 1, eos: 2
#             for i in range(len(trg)):
#                 idx_yp_i = torch.where(trg[i] == 2)[0]
#                 if len(idx_yp_i) > 0:
#                     if len(trg[i]) > idx_yp_i[0]+1:
#                         trg[i][idx_yp_i[0]+1:] = 0

#             src_mask = src == 0
#             trg_mask = trg == 0

#             log_prob = self.forward(
#                 src, src_mask, trg[:, :-1], trg_mask[:, :-1], z
#             )
#             log_prob = log_prob.transpose(1, 2)
#             nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
#             nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
#             return nll

#     def nll_train(self, src, trg, z=None):
#         # remove all the tokens
#         # after the first eos
#         # pad: 0, bos: 1, eos: 2
#         for i in range(len(trg)):
#             idx_yp_i = torch.where(trg[i] == 2)[0]
#             if len(idx_yp_i) > 0:
#                 if len(trg[i]) > idx_yp_i[0]+1:
#                     trg[i][idx_yp_i[0]+1:] = 0

#         src_mask = src == 0
#         trg_mask = trg == 0

#         log_prob = self.forward(
#             src, src_mask, trg[:, :-1], trg_mask[:, :-1], z
#         )
#         log_prob = log_prob.transpose(1, 2)
#         nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
#         nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
#         return nll


#     def get_params(self):
#         return self.h_params

#     @classmethod
#     def load(cls, path):
#         state = torch.load(path)
#         model = cls(**state["h_params"])
#         model.load_state_dict(state["weights"])
#         return model

#     # def get_src_enc(self, src, src_mask):
#     #     src_m = self.compute_padding_mask(src_mask)
#     #     src_emb = self.pe(self.token_emb(src))
#     #     src_enc = self.transformer.encoder(src_emb, src_m)
#     #     return src_enc


#     # def get_z_beamsearch(self, src_enc, src_mask, trg, trg_mask):

#     #     trg_m = self.compute_padding_mask(trg_mask)
#     #     trg_emb = self.pe(self.token_emb(trg))
#     #     trg_enc = self.transformer.encoder(trg_emb, trg_m)

#     #     z_src_den = torch.sum(1 - src_mask.float(), dim=1, keepdims=True)
#     #     z_src_den = torch.maximum(z_src_den, torch.ones_like(z_src_den))
#     #     z_src = torch.sum(src_enc, dim=1) / z_src_den

#     #     z_trg_den = torch.sum(1 - trg_mask.float(), dim=1, keepdims=True)
#     #     z_trg_den = torch.maximum(z_trg_den, torch.ones_like(z_trg_den))
#     #     z_trg = torch.sum(trg_enc, dim=1) / z_trg_den

#     #     mu_diff = self.interpolate_f(z_trg, z_src)

#     #     z = self.proj(mu_diff)

#     #     return z, mu_diff


#     def get_z(self, src, src_mask, trg, trg_mask):

#         src_m = self.compute_padding_mask(src_mask)
#         src_emb = self.pe(self.token_emb(src))
#         src_enc = self.transformer.encoder(src_emb, src_m)

#         trg_m = self.compute_padding_mask(trg_mask)
#         trg_emb = self.pe(self.token_emb(trg))
#         trg_enc = self.transformer.encoder(trg_emb, trg_m)

#         z_src_den = torch.sum(1 - src_mask.float(), dim=1, keepdims=True)
#         z_src_den = torch.maximum(z_src_den, torch.ones_like(z_src_den))
#         z_src = torch.sum(src_enc, dim=1) / z_src_den

#         z_trg_den = torch.sum(1 - trg_mask.float(), dim=1, keepdims=True)
#         z_trg_den = torch.maximum(z_trg_den, torch.ones_like(z_trg_den))
#         z_trg = torch.sum(trg_enc, dim=1) / z_trg_den

#         mu_diff = self.interpolate_f(z_trg, z_src)

#         z = self.proj(mu_diff)

#         return z, mu_diff

#     def encode(self, src, src_mask):
#         z = torch.randn(len(src), 16).to(src.device)
#         src_m = self.compute_padding_mask(src_mask)
#         src_emb = self.pe(self.token_emb(src))
#         src_enc = self.transformer.encoder(src_emb, src_m)
#         src_enc = src_enc + self.inv_proj(z).unsqueeze(1) * (1 - src_mask.float().unsqueeze(-1))
#         return src_enc

#     def decode(self, src_enc, trg, trg_mask):
#         trg_emb = self.pe(self.token_emb(trg))
#         trg_m = self.compute_padding_mask(trg_mask)
#         trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))
#         y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
#         y = torch.log_softmax(self.generator(y), dim=-1)
#         return y

#     def forward(self, src, src_mask, trg, trg_mask, z):
#         src_m = self.compute_padding_mask(src_mask)
#         trg_m = self.compute_padding_mask(trg_mask)

#         trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))

#         src_emb = self.pe(self.token_emb(src))
#         trg_emb = self.pe(self.token_emb(trg))
#         src_enc = self.transformer.encoder(src_emb, src_m)

#         if z is None:
#             z, mu_diff = self.get_z(src, src_mask, trg, trg_mask)

#         src_enc = src_enc + self.inv_proj(z).unsqueeze(1) * (1 - src_mask.float().unsqueeze(-1))

#         y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)

#         y = torch.log_softmax(self.generator(y), dim=-1)
#         return y

# prior bilinear b2
# class _One2Many(torch.nn.Module):
#     def __init__(
#         self,
#         *,
#         vocabulary_size,
#         d_model=256,
#         num_heads=8,
#         num_layers=6,
#         dim_feedforward=2048,
#         dropout=0.1,
#         layer_norm_eps=1e-6,
#         hidden_ae_dim=2
#     ):
#         super().__init__()
#         _locals = locals()
#         self.h_params = {}
#         for k in _locals:
#             if k == "self" or k.startswith("_"):
#                 continue
#             setattr(self, k, _locals[k])
#             self.h_params[k] = _locals[k]

#         self.token_emb = torch.nn.Embedding(vocabulary_size, d_model)
#         self.pe = PositionalEncoding(d_model, dropout)

#         self.transformer = torch.nn.Transformer(
#             d_model=d_model,
#             nhead=num_heads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             layer_norm_eps=layer_norm_eps,
#             batch_first=True,
#         )
#         self.generator = torch.nn.Linear(d_model, vocabulary_size)

#         self.proj = torch.nn.Sequential(
#             torch.nn.Linear(d_model, d_model//2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(d_model//2, d_model//4),
#             torch.nn.ReLU(),
#             torch.nn.Linear(d_model//4, hidden_ae_dim),
#         )

#         layer_norm_kwargs = {"elementwise_affine": False}
#         if int(torch.__version__[0]) >= 2:
#             layer_norm_kwargs["bias"] = False

#         self.interpolate_f = torch.nn.Bilinear(d_model, d_model, d_model, bias=False)

#         self.inv_proj = torch.nn.Sequential(
#             torch.nn.Linear(hidden_ae_dim, d_model//4),
#             torch.nn.ReLU(),
#             torch.nn.Linear(d_model//4, d_model//2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(d_model//2, d_model),
#             torch.nn.LayerNorm(d_model, **layer_norm_kwargs),
#         )
#         self.current_z = None

#     def compute_padding_mask(self, mask):
#         """
#         0 means attend / 1 means ignore
#         PyTorch wants a tensor of size B*H x N x N
#         where B is the batch size, H is the number of heads, N is
#         the sequence length
#         """
#         pmask = mask.unsqueeze(1)
#         pmask = pmask.expand(len(pmask), pmask.shape[-1], pmask.shape[-1])
#         pmask = pmask.unsqueeze(1)
#         pmask = pmask.expand(
#             len(pmask), self.num_heads, pmask.shape[-1], pmask.shape[-1]
#         )
#         pmask = pmask.reshape((-1, pmask.shape[-1], pmask.shape[-1]))
#         return pmask.to(mask.device)

#     def compute_causal_mask(self, mask):
#         attn_shape = (1, mask.shape[-1], mask.shape[-1])
#         cmask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
#         return cmask.to(mask.device)

#     def save(self, path):
#         state = {"h_params": self.h_params, "weights": self.state_dict()}
#         torch.save(state, path)

#     def nll(self, src, trg, z=None):
#         with torch.no_grad():
#             self.eval()
#             # remove all the tokens
#             # after the first eos
#             # pad: 0, bos: 1, eos: 2
#             for i in range(len(trg)):
#                 idx_yp_i = torch.where(trg[i] == 2)[0]
#                 if len(idx_yp_i) > 0:
#                     if len(trg[i]) > idx_yp_i[0]+1:
#                         trg[i][idx_yp_i[0]+1:] = 0

#             src_mask = src == 0
#             trg_mask = trg == 0

#             log_prob = self.forward(
#                 src, src_mask, trg[:, :-1], trg_mask[:, :-1], z
#             )
#             log_prob = log_prob.transpose(1, 2)
#             nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
#             nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
#             return nll

#     def nll_train(self, src, trg, z=None):
#         # remove all the tokens
#         # after the first eos
#         # pad: 0, bos: 1, eos: 2
#         for i in range(len(trg)):
#             idx_yp_i = torch.where(trg[i] == 2)[0]
#             if len(idx_yp_i) > 0:
#                 if len(trg[i]) > idx_yp_i[0]+1:
#                     trg[i][idx_yp_i[0]+1:] = 0

#         src_mask = src == 0
#         trg_mask = trg == 0

#         log_prob = self.forward(
#             src, src_mask, trg[:, :-1], trg_mask[:, :-1], z
#         )
#         log_prob = log_prob.transpose(1, 2)
#         nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
#         nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
#         return nll


#     def get_params(self):
#         return self.h_params

#     @classmethod
#     def load(cls, path):
#         state = torch.load(path)
#         model = cls(**state["h_params"])
#         model.load_state_dict(state["weights"])
#         return model

#     # def get_src_enc(self, src, src_mask):
#     #     src_m = self.compute_padding_mask(src_mask)
#     #     src_emb = self.pe(self.token_emb(src))
#     #     src_enc = self.transformer.encoder(src_emb, src_m)
#     #     return src_enc


#     # def get_z_beamsearch(self, src_enc, src_mask, trg, trg_mask):

#     #     trg_m = self.compute_padding_mask(trg_mask)
#     #     trg_emb = self.pe(self.token_emb(trg))
#     #     trg_enc = self.transformer.encoder(trg_emb, trg_m)

#     #     z_src_den = torch.sum(1 - src_mask.float(), dim=1, keepdims=True)
#     #     z_src_den = torch.maximum(z_src_den, torch.ones_like(z_src_den))
#     #     z_src = torch.sum(src_enc, dim=1) / z_src_den

#     #     z_trg_den = torch.sum(1 - trg_mask.float(), dim=1, keepdims=True)
#     #     z_trg_den = torch.maximum(z_trg_den, torch.ones_like(z_trg_den))
#     #     z_trg = torch.sum(trg_enc, dim=1) / z_trg_den

#     #     mu_diff = self.interpolate_f(z_trg, z_src)

#     #     z = self.proj(mu_diff)

#     #     return z, mu_diff


#     def get_z(self, src, src_mask, trg, trg_mask):

#         src_m = self.compute_padding_mask(src_mask)
#         src_emb = self.pe(self.token_emb(src))
#         src_enc = self.transformer.encoder(src_emb, src_m)
#         src_enc = src_enc * ( 1 - src_mask.float()[...,None] )

#         trg_m = self.compute_padding_mask(trg_mask)
#         trg_emb = self.pe(self.token_emb(trg))
#         trg_enc = self.transformer.encoder(trg_emb, trg_m)
#         trg_enc = trg_enc * ( 1 - trg_mask.float()[...,None] )

#         z_src_den = torch.sum(1 - src_mask.float(), dim=1, keepdims=True)
#         z_src_den = torch.maximum(z_src_den, torch.ones_like(z_src_den))
#         z_src = torch.sum(src_enc, dim=1) / z_src_den

#         z_trg_den = torch.sum(1 - trg_mask.float(), dim=1, keepdims=True)
#         z_trg_den = torch.maximum(z_trg_den, torch.ones_like(z_trg_den))
#         z_trg = torch.sum(trg_enc, dim=1) / z_trg_den

#         mu_diff = self.interpolate_f(z_trg, z_src)

#         z = self.proj(mu_diff)

#         return z, mu_diff

#     def encode(self, src, src_mask):
#         z = torch.randn(len(src), 2).to(src.device)
#         src_m = self.compute_padding_mask(src_mask)
#         src_emb = self.pe(self.token_emb(src))
#         src_enc = self.transformer.encoder(src_emb, src_m)
#         src_enc = src_enc * ( 1 - src_mask.float()[...,None] )
#         src_enc = src_enc + self.inv_proj(z).unsqueeze(1) * (1 - src_mask.float().unsqueeze(-1))
#         return src_enc

#     def decode(self, src_enc, trg, trg_mask):
#         trg_emb = self.pe(self.token_emb(trg))
#         trg_m = self.compute_padding_mask(trg_mask)
#         trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))
#         y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
#         y = torch.log_softmax(self.generator(y), dim=-1)
#         return y

#     def forward(self, src, src_mask, trg, trg_mask, z):
#         src_m = self.compute_padding_mask(src_mask)
#         trg_m = self.compute_padding_mask(trg_mask)

#         trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))

#         src_emb = self.pe(self.token_emb(src))
#         trg_emb = self.pe(self.token_emb(trg))
#         src_enc = self.transformer.encoder(src_emb, src_m)
#         src_enc = src_enc * ( 1 - src_mask.float()[...,None] )

#         if z is None:
#             z, mu_diff = self.get_z(src, src_mask, trg, trg_mask)

#         src_enc = src_enc + self.inv_proj(z).unsqueeze(1) * (1 - src_mask.float().unsqueeze(-1))

#         y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)

#         y = torch.log_softmax(self.generator(y), dim=-1)
#         return y


class _One2Many(torch.nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-6,
        hidden_ae_dim=4,
    ):
        super().__init__()
        _locals = locals()
        self.h_params = {}
        for k in _locals:
            if k == "self" or k.startswith("_"):
                continue
            setattr(self, k, _locals[k])
            self.h_params[k] = _locals[k]

        self.token_emb = torch.nn.Embedding(vocabulary_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self.generator = torch.nn.Linear(d_model, vocabulary_size)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, d_model // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 4, hidden_ae_dim),
        )

        self.interpolate_f = torch.nn.Linear(d_model * 2, d_model, bias=False)

        self.inv_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_ae_dim, d_model // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 4, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, d_model),
        )
        self.current_z = None
        self.hidden_ae_dim = hidden_ae_dim

    def init_z(self, batch_size, device):
        self.current_z = torch.randn(batch_size, self.hidden_ae_dim).to(device)

    def compute_padding_mask(self, mask):
        """
        0 means attend / 1 means ignore
        PyTorch wants a tensor of size B*H x N x N
        where B is the batch size, H is the number of heads, N is
        the sequence length
        """
        pmask = mask.unsqueeze(1)
        pmask = pmask.expand(len(pmask), pmask.shape[-1], pmask.shape[-1])
        pmask = pmask.unsqueeze(1)
        pmask = pmask.expand(len(pmask), self.num_heads, pmask.shape[-1], pmask.shape[-1])
        pmask = pmask.reshape((-1, pmask.shape[-1], pmask.shape[-1]))
        return pmask.to(mask.device)

    def compute_causal_mask(self, mask):
        attn_shape = (1, mask.shape[-1], mask.shape[-1])
        cmask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
        return cmask.to(mask.device)

    def save(self, path):
        state = {"h_params": self.h_params, "weights": self.state_dict()}
        torch.save(state, path)

    def nll(self, src, trg, z=None):
        with torch.no_grad():
            self.eval()
            # remove all the tokens
            # after the first eos
            # pad: 0, bos: 1, eos: 2
            for i in range(len(trg)):
                idx_yp_i = torch.where(trg[i] == 2)[0]
                if len(idx_yp_i) > 0:
                    if len(trg[i]) > idx_yp_i[0] + 1:
                        trg[i][idx_yp_i[0] + 1 :] = 0

            src_mask = src == 0
            trg_mask = trg == 0

            log_prob = self.forward(src, src_mask, trg[:, :-1], trg_mask[:, :-1], z)
            log_prob = log_prob.transpose(1, 2)
            nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
            nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
            return nll

    def nll_train(self, src, trg, z=None):
        # remove all the tokens
        # after the first eos
        # pad: 0, bos: 1, eos: 2
        for i in range(len(trg)):
            idx_yp_i = torch.where(trg[i] == 2)[0]
            if len(idx_yp_i) > 0:
                if len(trg[i]) > idx_yp_i[0] + 1:
                    trg[i][idx_yp_i[0] + 1 :] = 0

        src_mask = src == 0
        trg_mask = trg == 0

        log_prob = self.forward(src, src_mask, trg[:, :-1], trg_mask[:, :-1], z)
        log_prob = log_prob.transpose(1, 2)
        nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
        nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
        return nll

    def get_params(self):
        return self.h_params

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        model = cls(**state["h_params"])
        model.load_state_dict(state["weights"])
        return model


    def get_z(self, src, src_mask, trg, trg_mask):
        src_m = self.compute_padding_mask(src_mask)
        src_emb = self.pe(self.token_emb(src))
        src_enc = self.transformer.encoder(src_emb, src_m)
        src_enc = src_enc * (1 - src_mask.float()[..., None])

        trg_m = self.compute_padding_mask(trg_mask)
        trg_emb = self.pe(self.token_emb(trg))
        trg_enc = self.transformer.encoder(trg_emb, trg_m)
        trg_enc = trg_enc * (1 - trg_mask.float()[..., None])

        z_src_den = torch.sum(1 - src_mask.float(), dim=1, keepdims=True)
        z_src_den = torch.maximum(z_src_den, torch.ones_like(z_src_den))
        z_src = torch.sum(src_enc, dim=1) / z_src_den

        z_trg_den = torch.sum(1 - trg_mask.float(), dim=1, keepdims=True)
        z_trg_den = torch.maximum(z_trg_den, torch.ones_like(z_trg_den))
        z_trg = torch.sum(trg_enc, dim=1) / z_trg_den

        z_src_trg = torch.cat([z_src, z_trg], dim=1)
        z_src_src = torch.cat([z_src, z_src], dim=1)
        
        z_t_s = self.proj(self.interpolate_f(z_src_trg))
        z_s_s = self.proj(self.interpolate_f(z_src_src))
        
        return z_t_s, z_s_s

    def encode(self, src, src_mask):
        z_t_s = torch.randn(len(src), self.hidden_ae_dim).to(src.device)
        z_s_s = torch.zeros_like(z_t_s)
        src_m = self.compute_padding_mask(src_mask)
        src_emb = self.pe(self.token_emb(src))
        src_enc = self.transformer.encoder(src_emb, src_m)
        src_enc = src_enc + (self.inv_proj(z_t_s).unsqueeze(1) - self.inv_proj(z_s_s).unsqueeze(1))# * (1 - src_mask.float().unsqueeze(-1))
        return src_enc, [z_t_s, z_s_s]

    def decode(self, src_enc, trg, trg_mask):
        trg_emb = self.pe(self.token_emb(trg))
        trg_m = self.compute_padding_mask(trg_mask)
        trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))
        y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
        y = torch.log_softmax(self.generator(y), dim=-1)
        return y

    def forward(self, src, src_mask, trg, trg_mask, z=None):
        z_t_s, z_s_s = z
        src_m = self.compute_padding_mask(src_mask)
        trg_m = self.compute_padding_mask(trg_mask)

        trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))

        src_emb = self.pe(self.token_emb(src))
        trg_emb = self.pe(self.token_emb(trg))
        src_enc = self.transformer.encoder(src_emb, src_m)

        src_enc = src_enc + (self.inv_proj(z_t_s).unsqueeze(1) - self.inv_proj(z_s_s).unsqueeze(1)) #* (1 - src_mask.float().unsqueeze(-1))

        y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
        y = torch.log_softmax(self.generator(y), dim=-1)
        return y


class One2Many:
    _model_type = "One2Many"
    _version = 2

    def __init__(self, vocabulary, network, max_sequence_length, device):
        self.vocabulary = vocabulary
        self.network = network
        self.device = device
        self.max_sequence_length = max_sequence_length
        self._sampling_modes_enum = SamplingModesEnum()
        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)
        self.tokenizer = SMILESTokenizer()

    @classmethod
    def create_from_dict(cls, save_dict: dict, mode: str, device: torch.device):
        model_type = save_dict.get("model_type")

        if model_type and model_type != cls._model_type:
            raise RuntimeError(f"Wrong type: {model_type} but expected {cls._model_type}")

        network = _One2Many(**save_dict["network_parameter"])
        network.load_state_dict(save_dict["network_state"])
        if mode == "training":
            network = network.train().to(device)
        else:
            network = network.eval().to(device)
        vocabulary = Vocabulary.load_from_dictionary(save_dict["vocabulary"])

        model = cls(vocabulary, network, save_dict["max_sequence_length"], device)
        return model

    def get_network_parameters(self):
        return self.network.parameters()

    def set_mode(self, mode: str):
        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def load_from_file(cls, file_path: str, mode: str, device: torch.device):
        """
        Loads a model from a single file
        :param file_path: Path to the saved model
        :return: An instance of the network
        """

        save_dict = torch.load(file_path, map_location=device)
        return cls.create_from_dict(save_dict, mode, device)

    def get_save_dict(self):
        """Return the layout of the save dictionary"""

        save_dict = dict(
            model_type=self._model_type,
            version=self._version,
            metadata=self.meta_data,
            vocabulary=self.vocabulary,
            max_sequence_length=self.max_sequence_length,
            network_parameter=self.network.get_params(),
            network_state=self.network.state_dict(),
        )

    @torch.no_grad()
    def sample(self, src, src_mask, decode_type) -> Tuple[List[str], List[str], List[float]]:
        """
        Sample molecules
        :param src: (batch, seq) A batch of padded input sequences.
        :param src_mask: (batch, seq, seq) Mask of the input sequences.
        :param decode_type: decode type
        """

        if not decode_type:
            decode_type = self._sampling_modes_enum.MULTINOMIAL
        if decode_type == self._sampling_modes_enum.BEAMSEARCH:
            raise ValueError("Beam search not implemented")

        else:
            src_mask = src == 0
            batch_size = src.shape[0]
            ys = torch.ones((batch_size, 1), dtype=src.dtype, device=self.device)

            encoder_outputs, z = self.network.encode(src, src_mask)
            nlls = torch.zeros(batch_size).to(self.device)

            break_mask = torch.zeros((len(src), 1), dtype=torch.bool, device=self.device)

            # FIXME: end_token = self.vocabulary.end_token
            end_token = self.vocabulary["$"]
            for i in range(self.max_sequence_length - 1):
                log_prob = self.network.decode(
                    encoder_outputs,
                    ys,
                    ys == 0,
                )
                # (batch, seq, voc) need to exclude the probability of the start token "1"
                prob = torch.softmax(log_prob[:, -1] / self.temperature, dim=-1)
                if decode_type == self._sampling_modes_enum.GREEDY:
                    raise ValueError("Greedy not implemented")
                elif decode_type == self._sampling_modes_enum.MULTINOMIAL:
                    next = torch.multinomial(prob, 1) * ~break_mask
                    ys = torch.cat((ys, next), -1)
                    nlls += self._nll_loss(log_prob[:, -1], next[:, 0])

                    break_mask = torch.logical_or(break_mask, next == 2)

                if torch.all(break_mask):
                    break

            tokenizer = self.tokenizer
            input_smiles_list = [
                tokenizer.untokenize(self.vocabulary.decode(seq))
                for seq in src.detach().cpu().numpy()
            ]
            output_smiles_list = [
                tokenizer.untokenize(self.vocabulary.decode(seq))
                for seq in ys.detach().cpu().numpy()
            ]
            nlls = nlls.detach().cpu().numpy()
            
            z_t_s, z_s_s = z

        return input_smiles_list, output_smiles_list, nlls, z_t_s, z_s_s

    def likelihood(self, src, src_mask, trg, trg_mask, z=None):
        """
        Retrieves the likelihood of molecules.
        :param src: (batch, seq) A batch of padded input sequences.
        :param src_mask: (batch, seq, seq) Mask of the input sequences.
        :param trg: (batch, seq) A batch of output sequences; with start token, without end token.
        :param trg_mask: Mask of the input sequences.
        :return:  (batch) Log likelihood for each output sequence in the batch.
        """

        # ifself.network.training:
        # trg_y = trg[:, 1:]  # skip start token but keep end token
        # trg = trg[:, :-1]  # save start token, skip end token
        # z2 = torch.randn_like(z)
        # log_prob = self.network.forward(src, src == 0, trg, trg == 0, z)
        # log_prob = log_prob.transpose(1, 2)  # (batch, voc, seq_len)
        # penalty = imq_kernel(z2, z) * 40.0
        # nll = self._nll_loss(log_prob, trg_y).sum(dim=1) + penalty
        #     if z_gt is None:
        #         z_gt = torch.randn_like(z)
        #     log_prob = self.network.forward(src, src==0, trg, trg==0, z)
        #     log_prob = log_prob.transpose( 1, 2 )  # (batch, voc, seq_len)
        #     # penalty = imq_kernel(z1, z) * 60.0
        #     # penalty = imq_kernel(z_gt, z) * 40.0
        #     # penalty = imq_kernel(z_gt, z) * 80.0
        #     # penalty = imq_kernel(z1, z2) * 20.0
        #     # penalty = imq_kernel(z1, z2) * 100.0
        #     # penalty = imq_kernel(z1, z2)
        #     nll = self._nll_loss(log_prob, trg_y).sum(dim=1) # + penalty
        #     # norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        
        if True:
            assert z is not None
            z_t_s, z_s_s  = self.network.get_z(src, src==0, trg, trg==0)
            z = [z_t_s, z_s_s]
            # z = [z_t_s, torch.zeros_like(z_t_s)]
            trg_y = trg[:, 1:]  # skip start token but keep end token
            trg = trg[:, :-1]  # save start token, skip end token

            log_prob = self.network.forward(src, src == 0, trg, trg == 0, z)
            log_prob = log_prob.transpose(1, 2)  # (batch, voc, seq_len)
            nll = self._nll_loss(log_prob, trg_y).sum(dim=1)
        return nll, z

    def set_temperature(self, t):
        self.temperature = t

    #     return save_dict

    # def save(self, path_to_file):
    #     """
    #     Saves the model to a file.
    #     :param path_to_file: Path to the file which the model will be saved to.
    #     """

    #     save_dict = self.get_save_dict()

    #     torch.save(save_dict, path_to_file)

    # save_to_file = save  # alias for backwards compatibility

    # def likelihood(self, warheads_seqs, warheads_seq_lengths, linker_seqs, linker_seq_lengths):
    #     """
    #     Retrieves the likelihood of warheads and their respective linker.
    #     :param warheads_seqs: (batch, seq) A batch of padded scaffold sequences.
    #     :param warheads_seq_lengths: The length of the scaffold sequences (for packing purposes).
    #     :param linker_seqs: (batch, seq) A batch of decorator sequences.
    #     :param linker_seq_lengths: The length of the decorator sequences (for packing purposes).
    #     :return:  (batch) Log likelihood for each item in the batch.
    #     """

    #     # NOTE: the decoration_seq_lengths have a - 1 to prevent the end token to be forward-passed.
    #     logits = self.network(
    #         warheads_seqs, warheads_seq_lengths, linker_seqs, linker_seq_lengths - 1
    #     )  # (batch, seq - 1, voc)
    #     log_probs = logits.log_softmax(dim=2).transpose(1, 2)  # (batch, voc, seq - 1)
    #     return self._nll_loss(log_probs, linker_seqs[:, 1:]).sum(dim=1)  # (batch)

    # @torch.no_grad()
    # def sample(self, inputs, input_seq_lengths) -> Tuple[List[str], List[str], List[float]]:
    #     """
    #     Samples as many linker as warhead pairs in the tensor.
    #     :param inputs: A tensor with the warheads to sample already encoded and padded.
    #     :param input_seq_lengths: A tensor with the length of the warheads.
    #     :return: a sampled sequence dto with input_smi, output_smi and nll
    #     """
    #     batch_size = inputs.size(0)

    #     input_vector = torch.full(
    #         (batch_size, 1), self.vocabulary.target.vocabulary["^"], dtype=torch.long
    #     )  # (batch, 1)
    #     seq_lengths = torch.ones(batch_size)  # (batch)
    #     encoder_padded_seqs, hidden_states = self.network.forward_encoder(inputs, input_seq_lengths)
    #     nlls = torch.zeros(batch_size)
    #     not_finished = torch.ones(batch_size, 1, dtype=torch.long)
    #     sequences = []
    #     for _ in range(self.max_sequence_length - 1):
    #         logits, hidden_states, _ = self.network.forward_decoder(
    #             input_vector, seq_lengths, encoder_padded_seqs, hidden_states
    #         )  # (batch, 1, voc)
    #         probs = logits.softmax(dim=2).squeeze(dim=1)  # (batch, voc)
    #         log_probs = logits.log_softmax(dim=2).squeeze(dim=1)  # (batch, voc)
    #         input_vector = torch.multinomial(probs, 1) * not_finished  # (batch, 1)
    #         sequences.append(input_vector)
    #         nlls += self._nll_loss(log_probs, input_vector.squeeze(dim=1))
    #         not_finished = (input_vector > 1).type(torch.long)  # 0 is padding, 1 is end token
    #         if not_finished.sum() == 0:
    #             break

    #     linker_smiles_list = [
    #         self.vocabulary.target.decode(seq) for seq in torch.cat(sequences, 1).data.cpu().numpy()
    #     ]
    #     warheads_smiles_list = [
    #         self.vocabulary.input.decode(seq) for seq in inputs.data.cpu().numpy()
    #     ]

    #     return (
    #         warheads_smiles_list,
    #         linker_smiles_list,
    #         nlls.data.cpu().numpy(),
    #     )

    # def get_network_parameters(self):
    #
