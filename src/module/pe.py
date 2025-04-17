import torch
import numpy as np

class PositionalEncoding(torch.nn.Module):
    """Implements the sin and cos positional encoding.
    Source: https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): the number of expected features in the input.
        dropout (float): the dropout value.
        max_len (int): the maximum length of the input sequence.
        
    Returns:
        torch.Tensor: input + positional encodings.
    
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)