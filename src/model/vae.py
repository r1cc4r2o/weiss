import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from src.module.pe import PositionalEncoding


class Mol2MolVAE(torch.nn.Module):
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
        hidden_ae_dim: int = 16,
        pooling_strategy: str = 'mean'
    ):
        super().__init__()
        """ Mol2Mol Transformer model with VAE.
        
        Args:
            vocabulary_size (int): the size of the vocabulary.
            d_model (int): the number of expected features in the input.
            num_heads (int): the number of heads in the multiheadattention models.
            num_layers (int): the number of encoder and decoder layers.
            dim_feedforward (int): the dimension of the feedforward network model.
            dropout (float): the dropout value.
            layer_norm_eps (float): the eps value in layer norm.
            hidden_ae_dim (int): the latent dimension.
            pooling_strategy (str): the pooling strategy.
            
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
            
        # Transformer
        
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
        
        # Variational Autoencoder
        
        self.hidden_ae_dim = hidden_ae_dim
        self.mu = nn.Linear(d_model, hidden_ae_dim, bias=False)
        self.logvar = nn.Linear(d_model, hidden_ae_dim, bias=False)
        self.memory_projection = nn.Linear(
            hidden_ae_dim,
            d_model,
            bias=False,
        )
        self.min_z = 0.5
        self.pooling_strategy = pooling_strategy
        

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
        """ Save the model to a file. """
        state = {"h_params": self.h_params, "weights": self.state_dict()}
        torch.save(state, path)

    def nll(self, src: torch.Tensor, trg: torch.Tensor, z: torch.Tensor):
        """ Compute the negative log likelihood.
        
        Args:
            src (torch.Tensor): the source tensor.  
            trg (torch.Tensor): the target tensor.
            z (torch.Tensor): the latent tensor.
            
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
            
            log_prob, mu, logvar = self.forward(
                src, src_mask, trg[:, :-1], trg_mask[:, :-1], z
            )
            log_prob = log_prob.transpose(1, 2)
            nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
            nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
            return nll

    @classmethod
    def load(cls, path: str):
        """Load the model from a file."""
        state = torch.load(path)
        model = cls(**state["h_params"])
        model.load_state_dict(state["weights"])
        return model


    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, trg: torch.Tensor, trg_mask: torch.Tensor, z=None):
        """ Forward pass.
        
        Args:
            src (torch.Tensor): the source tensor.
            src_mask (torch.Tensor): the source mask.
            trg (torch.Tensor): the target tensor.
            trg_mask (torch.Tensor): the target mask.
            z (torch.Tensor): the latent tensor.
            
        Returns:
            torch.Tensor: the output tensor.
            torch.Tensor: the mean tensor.
            torch.Tensor: the logvar tensor.
        
        """
        
        src_m = self.compute_padding_mask(src_mask)
        trg_m = self.compute_padding_mask(trg_mask)

        trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))

        src_emb = self.pe(self.token_emb(src))
        trg_emb = self.pe(self.token_emb(trg))
        src_enc = self.transformer.encoder(src_emb, src_m)
        src_enc = src_enc * ( 1 - src_mask.float()[...,None] )
        
        if z is None:
            pooled = self.avg_pooling(src_enc, src_mask) # B x D
            z, mu, logvar = self.calculate_latent(pooled) # B x H (H: hidden_ae_dim)
        else:
            mu = None
            logvar = None
            
        src_enc = self.build_past(z) # B x 1 x D

        y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
        y = torch.log_softmax(self.generator(y), dim=-1)
        
        return y, mu, logvar
    
    def decode(self, src_enc: torch.Tensor, trg: torch.Tensor, trg_mask: torch.Tensor):
        """ Decode the target tensor.
        
        Args:
            src_enc (torch.Tensor): the source encoding tensor.
            trg (torch.Tensor): the target tensor.
            trg_mask (torch.Tensor): the target mask tensor.
            
        Returns:
            torch.Tensor: the output tensor.
            
        """
        trg_m = self.compute_padding_mask(trg_mask)
        trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask)) 
        trg_emb = self.pe(self.token_emb(trg))
        y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
        y = torch.log_softmax(self.generator(y), dim=-1)
        return y

    def calculate_latent(self, pooled: torch.Tensor):
        """ Compute the latent vector.
        
        Args:
            pooled (torch.Tensor): the pooled source tensor.
            
        Returns:
            torch.Tensor: the latent tensor.
            torch.Tensor: the mean tensor.
            torch.Tensor: the logvar tensor.
            
        """
        mu, logvar = self.mu(pooled), self.logvar(pooled)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """ Reparameterization trick to sample from 
        N(mu, var) from N(0,1).
        
        Args:
            mu (torch.Tensor): the mean tensor [B x D].
            logvar (torch.Tensor): the logvar tensor [B x D].

        Returns:
            torch.Tensor: the reparameterized tensor [B x D].
        
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def build_past(self, z: torch.Tensor):
        """ Build the past key values.
        
        Args:
            z (torch.Tensor): the latent tensor.
            
        Returns:
            torch.Tensor: the past key values.
            
        """
        z = self.memory_projection(z) # B x D
        past_key_values = z.unsqueeze(1) # B x 1 x D
        return past_key_values
    
    def avg_pooling(self, enc: torch.Tensor, mask: torch.Tensor):
        """ Average pooling on the encoding of the source
        sequence.
        
        Args:
            enc (torch.Tensor): the encoding tensor.
            mask (torch.Tensor): the mask tensor.
            
        Returns:
            torch.Tensor: the pooled tensor.
            
        """
        z_den = torch.sum(1 - mask.float(), dim=1, keepdims=True)
        z_den = torch.maximum(z_den, torch.ones_like(z_den))
        enc = enc * ( 1 - mask.float()[...,None] )
        z_trg = torch.sum(enc, dim=1) / z_den
        return z_trg