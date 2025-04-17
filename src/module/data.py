# This file contains the Dataset class which is used to create a torch Dataset object from a list of data.

import torch
from typing import List


#####################################


class PairedDataset(torch.utils.data.Dataset):
    """Given a list of data, this class creates a torch Dataset object,
    where each element is a pair of data."""
    def __init__(self, data: List):
        self.data = data

    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        """ Collate function for the PairedDataset."""
        src, trg = zip(*batch)
        src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
        trg = torch.nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)
        return src, trg
    

#####################################


class SimpleDataloader(torch.utils.data.Dataset):
    """Given a list of data, this class creates a torch Dataset object."""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        """ Collate function for the SimpleDataloader."""
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        return batch
    
    
    
#####################################
