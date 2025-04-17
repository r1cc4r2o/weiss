 # This file contains the LSTM model for the Mol2Mol model.
 
import torch

class Mol2MolLSTM(torch.nn.Module):
    def __init__(self, *, 
        vocabulary_size, 
        d_model: int = 256, 
        num_layers: int = 6, 
        dropout: float = 0.1, 
        bidirectional: bool = False
        ):
        super().__init__()
        """ Mol2Mol model with LSTM.
        
        Args:
            vocabulary_size (int): the size of the vocabulary.
            d_model (int): the number of expected features in the input.
            num_layers (int): the number of encoder and decoder layers.
            dropout (float): the dropout value.
            bidirectional (bool): the bidirectional value.
            
        Returns:
            None

        """

        _locals = locals()
        self.hyper_parameters = {}
        for k in _locals:
            if k == "self" or k.startswith("_"):
                continue
            setattr(self, k, _locals[k])
            self.hyper_parameters[k] = _locals[k]

        self.src_emb = torch.nn.Embedding(vocabulary_size, d_model)
        self.trg_emb = torch.nn.Embedding(vocabulary_size, d_model)

        self.src_dropout = torch.nn.Dropout(dropout)
        self.trg_dropout = torch.nn.Dropout(dropout)

        self.encoder = torch.nn.LSTM(d_model, d_model, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.decoder = torch.nn.LSTM(d_model, d_model, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        
        self.generator = torch.nn.Linear(d_model, vocabulary_size)
        
        
    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the Mol2Mol model with LSTM.
        
        Args:
            src (torch.Tensor): the source tensor.
            trg (torch.Tensor): the target tensor.
            
        Returns:
            torch.Tensor: the output tensor.
        
        """
        
        # Embed the source and target tensors
        
        src_emb = self.src_dropout(self.src_emb(src)) # shape: (B, N, D)
        trg_emb = self.trg_dropout(self.trg_emb(trg)) # shape: (B, M, D)
        
        # Encode the source tensor
        
        _, enc_hid = self.encoder(src_emb)
        
        # Decode the target tensor
        
        trg_enc, _ = self.decoder(trg_emb, enc_hid)
        y = torch.log_softmax(self.generator(trg_enc), dim=-1)

        return y
    
    
    def decode(self, enc_hid: torch.Tensor, trg: torch.Tensor):
        """ Decode the target tensor.
        
        Args:
            enc_hid (torch.Tensor): the encoded hidden state.
            trg (torch.Tensor): the target tensor.
            
        Returns:
            torch.Tensor: the output tensor.
        
        """
        trg_emb = self.trg_dropout(self.trg_emb(trg)) # shape: (B, M, D)
        trg_enc, _ = self.decoder(trg_emb, enc_hid)
        y = torch.log_softmax(self.generator(trg_enc), dim=-1)
        return y
    
    
    def nll(self, src: torch.Tensor, trg: torch.Tensor):
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
                        
            log_prob = self(src, trg[:, :-1])
            
            log_prob = log_prob.transpose(1, 2)
            
            nll_loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)
            
            nll = nll_loss(log_prob, trg[:, 1:]).sum(dim=1)
            
        return nll
    
    
    