# This file contains the Transformer model for the Mol2Mol model.

import torch
import numpy as np

from src.module.pe import PositionalEncoding


class Mol2Mol(torch.nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        """ Mol2Mol Transformer model. 
        Source: 10.26434/chemrxiv-2024-r9ljm-v2
        
        Args:
            vocabulary_size (int): the size of the vocabulary.
            d_model (int): the number of expected features in the input.
            num_heads (int): the number of heads in the multiheadattention models.
            num_layers (int): the number of encoder and decoder layers.
            dim_feedforward (int): the dimension of the feedforward network model.
            dropout (float): the dropout value.
            layer_norm_eps (float): the eps value in layer norm.
            
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

       
    def compute_padding_mask(self, mask):
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

    def compute_causal_mask(self, mask):
        """ Compute the causal mask.
        
        Args:
            mask (torch.Tensor): the mask tensor.
            
        Returns:
            torch.Tensor: the causal mask.
            
        """
        attn_shape = (1, mask.shape[-1], mask.shape[-1])
        cmask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
        return cmask.to(mask.device)

    def save(self, path):
        """ Save the model to a file. """
        state = {"h_params": self.h_params, "weights": self.state_dict()}
        torch.save(state, path)

    def nll(self, src, trg):
        """ Compute the negative log likelihood.
        
        Args:
            src (torch.Tensor): the source tensor.  
            trg (torch.Tensor): the target tensor.
            
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
                src, src_mask, trg[:, :-1], trg_mask[:, :-1]
            )
            log_prob = log_prob.transpose(1, 2)
            nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
            nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
            return nll

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        model = cls(**state["h_params"])
        model.load_state_dict(state["weights"])
        return model
    
    def get_params(self):
        """Get the hyperparameters of the network."""
        return self.h_params
    
    # def get_src_enc(self, src, src_mask):
    #     src_m = self.compute_padding_mask(src_mask)
    #     src_emb = self.pe(self.token_emb(src))
    #     src_enc = self.transformer.encoder(src_emb, src_m)
    #     return src_enc

    def forward(self, src, src_mask, trg, trg_mask):
        """Forward pass of the model.
        
        Args:   
            src (torch.Tensor): the source tensor.
            src_mask (torch.Tensor): the source mask tensor.
            trg (torch.Tensor): the target tensor.
            trg_mask (torch.Tensor): the target mask tensor.
            
        Returns:
            torch.Tensor: the decoded target tensor.
            
        """
        src_m = self.compute_padding_mask(src_mask)
        trg_m = self.compute_padding_mask(trg_mask)

        trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))

        src_emb = self.pe(self.token_emb(src))
        trg_emb = self.pe(self.token_emb(trg))
        src_enc = self.transformer.encoder(src_emb, src_m)

        y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
        y = torch.log_softmax(self.generator(y), dim=-1)
        return y
    
    
    def encode(self, src, src_mask):
        """Encode the source tensor.
        
        Args:
            src (torch.Tensor): the source tensor.
            src_mask (torch.Tensor): the source mask tensor.
            
        Returns:
            torch.Tensor: the encoded source tensor.
            
        """
        src_m = self.compute_padding_mask(src_mask)
        src_emb = self.pe(self.token_emb(src))
        src_enc = self.transformer.encoder(src_emb, src_m)
        return src_enc, None
    
    def decode(self, src_enc, trg, trg_mask):
        """Decode the target tensor.
        
        Args:
            src_enc (torch.Tensor): the encoded source tensor.
            trg (torch.Tensor): the target tensor.
            trg_mask (torch.Tensor): the target mask tensor.
            
        Returns:
            torch.Tensor: the decoded target tensor.
            
        """
        trg_m = self.compute_padding_mask(trg_mask)
        trg_m_and_cm = torch.logical_or(trg_m, self.compute_causal_mask(trg_mask))
        trg_emb = self.pe(self.token_emb(trg))
        y = self.transformer.decoder(trg_emb, memory=src_enc, tgt_mask=trg_m_and_cm)
        y = torch.log_softmax(self.generator(y), dim=-1)
        return y