from __future__ import annotations

__all__ = ["SampleBatch", "SmilesState"]
from dataclasses import dataclass
from enum import Enum
from typing import List, TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    import numpy as np


class SmilesState(Enum):
    INVALID = 0
    VALID = 1
    DUPLICATE = 2


@dataclass
class BatchRow:
    input: str
    output: str
    nll: float
    smiles: str
    state: SmilesState
    
@dataclass
class BatchRowZ:
    input: str
    output: str
    nll: float
    smiles: str
    state: SmilesState
    z_t_s: Tensor
    z_s_s: Tensor


@dataclass
class SampleBatch:
    """Container to hold the data returned by the adapter .sample() methods

    This is a somewhat ugly unifying implementation for all generator sample
    methods which return different data.  All return a 3-tuple with the NLL last
    but Reinvent returns one SMILES list while the others
    return two SMILES lists.
    """

    items1: List[str] | None  # SMILES, None for Reinvent
    items2: List[str]  # SMILES
    nlls: Tensor  # negative log likelihoods from the model
    smilies: List[str] = None  # processed SMILES
    states: np.ndarray[SmilesState] = None  # states for items2

    def __post_init__(self):
        """Set various aliases for the fields"""

        # Libinvent
        self.scaffolds = self.items1
        self.decorations = self.items2

        # Linkinvent
        self.warheads = self.items1
        self.linkers = self.items2

        # Mol2Mol
        self.input = self.items1
        self.output = self.items2

        # loop counter for iterator
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.idx

        try:
            if self.smilies:
                smiles = self.smilies[idx]
            else:
                smiles = None

            if self.states is not None:
                state = self.states[idx]
            else:
                state = None

            result = BatchRow(
                self.items1[idx],
                self.items2[idx],
                self.nlls[idx],
                smiles,
                state,
            )
        except IndexError:
            self.idx = 0
            raise StopIteration

        self.idx += 1

        return result

    @classmethod
    def from_list(cls, batch_rows: List[BatchRow]) -> SampleBatch:
        """Create a new dataclass from a list of BatchRow

        This factory class requires a list with a 5-tuple for the 5 fields.
        This is needed for Libinvent, Linkinvent, Mol2mol.

        FIXME: data type consistency

        :param batch_rows: list of batch rows
        :returns: a new dataclass made from the list
        """

        combined = []

        for batch_row in batch_rows:
            combined.append(
                (
                    batch_row.input,
                    batch_row.output,
                    batch_row.nll,
                    batch_row.smiles,
                    batch_row.state,
                )
            )

        transpose = list(zip(*combined))

        assert len(transpose) == 5

        sample_batch = cls(*transpose)
        sample_batch.nlls = Tensor(sample_batch.nlls)

        return sample_batch



@dataclass
class SampleBatchZ:
    """Container to hold the data returned by the adapter .sample() methods

    This is a somewhat ugly unifying implementation for all generator sample
    methods which return different data.  All return a 3-tuple with the NLL last
    but Reinvent returns one SMILES list while the others
    return two SMILES lists.
    """

    items1: List[str] | None  # SMILES, None for Reinvent
    items2: List[str]  # SMILES
    nlls: Tensor  # negative log likelihoods from the model
    z_t_s: Tensor
    z_s_s: Tensor
    smilies: List[str] = None  # processed SMILES
    states: np.ndarray[SmilesState] = None  # states for items2
    

    def __post_init__(self):
        """Set various aliases for the fields"""

        # Libinvent
        self.scaffolds = self.items1
        self.decorations = self.items2

        # Linkinvent
        self.warheads = self.items1
        self.linkers = self.items2

        # Mol2Mol
        self.input = self.items1
        self.output = self.items2

        # loop counter for iterator
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.idx

        try:
            if self.smilies:
                smiles = self.smilies[idx]
            else:
                smiles = None

            if self.states is not None:
                state = self.states[idx]
            else:
                state = None
                
            # if self.z is not None:
            #     z = self.z[idx]
            # else:
            #     z = None

            result = BatchRowZ(
                self.items1[idx],
                self.items2[idx],
                self.nlls[idx],
                self.z_t_s[idx], # z_t_s
                self.z_s_s[idx], # z_s_s
                smiles,
                state,
            )
        except IndexError:
            self.idx = 0
            raise StopIteration

        self.idx += 1

        return result

    @classmethod
    def from_list(cls, batch_rows: List[BatchRowZ]) -> SampleBatchZ:
        """Create a new dataclass from a list of BatchRow

        This factory class requires a list with a 5-tuple for the 5 fields.
        This is needed for Libinvent, Linkinvent, Mol2mol.

        FIXME: data type consistency

        :param batch_rows: list of batch rows
        :returns: a new dataclass made from the list
        """

        combined = []

        for batch_row in batch_rows:
            combined.append(
                (
                    batch_row.input,
                    batch_row.output,
                    batch_row.nll,
                    batch_row.smiles,
                    batch_row.state,
                    batch_row.z_t_s, 
                    batch_row.z_s_s, 
                )
            )

        transpose = list(zip(*combined))

        assert len(transpose) == 7

        sample_batch = cls(*transpose)
        sample_batch.nlls = Tensor(sample_batch.nlls)
        # sample_batch.z = Tensor(sample_batch.z)
        

        return sample_batch