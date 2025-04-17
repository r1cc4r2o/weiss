# This file contains the implementation of the Mol2MolWEISS with the bias term.
# WEISS: Wasserstein Efficient Sampling Strategy for LLMs in Drug Design

import torch
import numpy as np
from typing import List

from src.module.pe import PositionalEncoding

class Mol2MolWEISSWithB(torch.nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        hidden_ae_dim: int = 16 
        
    ):   
        super().__init__()
        """ Original implementation of Mol2MolWEISS with Bias.
        WEISS: Wasserstein Efficient Sampling Strategy for LLMs in Drug Design
        
        
        Args:
            vocabulary_size (int): the size of the vocabulary.
            d_model (int): the number of expected features in the input.
            num_heads (int): the number of heads in the multiheadattention models.
            num_layers (int): the number of sub-encoder-layers in the encoder.
            dim_feedforward (int): the dimension of the feedforward network model.
            dropout (float): the dropout value.
            layer_norm_eps (float): the epsilon value for layer normalization.
            hidden_ae_dim (int): the dimension of the hidden autoencoder.
            
        Returns:
            None
        
        """
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
            torch.nn.Linear(d_model, d_model//2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model//2, d_model//4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model//4, hidden_ae_dim),
        )

        self.interpolate_f = torch.nn.Linear(d_model*2, d_model, bias=False)
        
        self.inv_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_ae_dim, d_model//4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model//4, d_model//2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model//2, d_model),
        )
        self.current_z = None
        self.hidden_ae_dim = hidden_ae_dim
        
    def init_z(self, batch_size: int, device: str):
        """Initialize the latent variable z."""
        self.current_z = torch.randn(batch_size, self.hidden_ae_dim).to(device)
       
    def compute_padding_mask(self, mask: torch.Tensor):
        """ Compute the padding mask.

        Args:
            mask (torch.Tensor): the mask tensor.
            
        Returns:
            torch.Tensor: the padding mask.
        
        0 means attend / 1 means ignore
        PyTorch wants a tensor of size B*H x N x N
        where B is the batch size, H is the number of heads, N is
        the sequence length
        
        """
        pmask = mask.unsqueeze(1)
        pmask = pmask.expand(len(pmask), pmask.shape[-1], pmask.shape[-1])
        pmask = pmask.unsqueeze(1)
        pmask = pmask.expand(
            len(pmask), self.num_heads, pmask.shape[-1], pmask.shape[-1]
        )
        pmask = pmask.reshape((-1, pmask.shape[-1], pmask.shape[-1]))
        return pmask.to(mask.device)

    def compute_causal_mask(self, mask: torch.Tensor):
        """ Compute the causal mask.
        
        Args:
            mask (torch.Tensor): the mask tensor.
            
        Returns:
            torch.Tensor: the causal mask.
            
        """
        attn_shape = (1, mask.shape[-1], mask.shape[-1])
        cmask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
        return cmask.to(mask.device)

    def save(self, path: str):
        state = {"h_params": self.h_params, "weights": self.state_dict()}
        torch.save(state, path)

    def nll(self, src: torch.Tensor, trg: torch.Tensor, z: List = None):
        """ Compute the negative log likelihood.
        
        Args:
            src (torch.Tensor): the source tensor.  
            trg (torch.Tensor): the target tensor.
            z (List): the latent tensor z = [z_t_s, z_s_s].
            
        Returns:
            torch.Tensor: the negative log likelihood.
            
        """
        with torch.no_grad():
            self.eval()
            # remove all the tokens
            # after the first eos
            # pad: 0, bos: 1, eos: 2
            for i in range(len(trg)):
                idx_yp_i = torch.where(trg[i] == 2)[0]
                if len(idx_yp_i) > 0:
                    if len(trg[i]) > idx_yp_i[0]+1:
                        trg[i][idx_yp_i[0]+1:] = 0
                    
            src_mask = src == 0
            trg_mask = trg == 0
            
            log_prob = self.forward(
                src, src_mask, trg[:, :-1], trg_mask[:, :-1], z
            )
            log_prob = log_prob.transpose(1, 2)
            nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
            nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
            return nll


    def get_params(self):
        """Get the hyperparameters of the network."""
        return self.h_params

    @classmethod
    def load(cls, path: str):
        """ Load the model from a file. """
        state = torch.load(path)
        model = cls(**state["h_params"])
        model.load_state_dict(state["weights"])
        return model

    
    def get_z(self, src: torch.Tensor, src_mask: torch.Tensor, trg: torch.Tensor, trg_mask: torch.Tensor):
        """ Compute the latent variable z given the source and target tensors.

        Args:
            src (torch.Tensor): the source tensor.
            src_mask (torch.Tensor): the source mask tensor.
            trg (torch.Tensor): the target tensor.
            trg_mask (torch.Tensor): the target mask tensor.
            
        Returns:
            torch.Tensor: the latent variable z projected.
            torch.Tensor: the average latent variable mu_diff.
            
        """
        src_m = self.compute_padding_mask(src_mask)
        src_emb = self.pe(self.token_emb(src))
        src_enc = self.transformer.encoder(src_emb, src_m)
        src_enc = src_enc * ( 1 - src_mask.float()[...,None] )
        trg_m = self.compute_padding_mask(trg_mask)
        trg_emb = self.pe(self.token_emb(trg))
        trg_enc = self.transformer.encoder(trg_emb, trg_m)
        trg_enc = trg_enc * ( 1 - trg_mask.float()[...,None] )
        
        z_src_den = torch.sum(1 - src_mask.float(), dim=1, keepdims=True)
        z_src_den = torch.maximum(z_src_den, torch.ones_like(z_src_den))
        z_src = torch.sum(src_enc, dim=1) / z_src_den
        z_trg_den = torch.sum(1 - trg_mask.float(), dim=1, keepdims=True)
        z_trg_den = torch.maximum(z_trg_den, torch.ones_like(z_trg_den))
        z_trg = torch.sum(trg_enc, dim=1) / z_trg_den

        z_src_trg = torch.cat([z_src, z_trg], dim=1)
        mu_diff = self.interpolate_f(z_src_trg)
        z = self.proj(mu_diff)
        return z, mu_diff

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        """Encode the source tensor.
        
        Args:
            src (torch.Tensor): the source tensor.
            src_mask (torch.Tensor): the source mask tensor.
            
        Returns:
            torch.Tensor: the encoded source tensor.
            torch.Tensor: the latent variable z.
            
        """
        z = torch.randn(len(src), self.hidden_ae_dim).to(src.device)
        src_m = self.compute_padding_mask(src_mask)
        src_emb = self.pe(self.token_emb(src))
        src_enc = self.transformer.encoder(src_emb, src_m)
        src_enc = src_enc + self.inv_proj(z).unsqueeze(1) 
        return src_enc, z


    def decode(self, src_enc: torch.Tensor, trg: torch.Tensor, trg_mask: torch.Tensor):
        """Decode the target tensor.
        
        Args:
            src_enc (torch.Tensor): the encoded source tensor.
            trg (torch.Tensor): the target tensor.
            trg_mask (torch.Tensor): the target mask tensor.
            
        Returns:
            torch.Tensor: the decoded target tensor.
            
        """
        trg_emb = self.pe(self.token_emb(trg))
        trg_m = self.compute_padding_mask(trg_mask)
        trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))
        y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
        y = torch.log_softmax(self.generator(y), dim=-1)
        return y

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, trg: torch.Tensor, trg_mask: torch.Tensor, z: List = None):
        """Forward pass of the model.
        
        Args:   
            src (torch.Tensor): the source tensor.
            src_mask (torch.Tensor): the source mask tensor.
            trg (torch.Tensor): the target tensor.
            trg_mask (torch.Tensor): the target mask tensor.
            z (torch.Tensor): the latent variable z.
            
        Returns:
            torch.Tensor: the decoded target tensor.
            
        """
        src_m = self.compute_padding_mask(src_mask)
        trg_m = self.compute_padding_mask(trg_mask)
        trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))
        src_emb = self.pe(self.token_emb(src))
        trg_emb = self.pe(self.token_emb(trg))
        src_enc = self.transformer.encoder(src_emb, src_m)
        if z is None:
            z, mu_diff = self.get_z(src, src_mask, trg, trg_mask)
        src_enc = src_enc + self.inv_proj(z).unsqueeze(1) 
        y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
        y = torch.log_softmax(self.generator(y), dim=-1)
        return y