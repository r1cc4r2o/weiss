"""Adapter for One2Many"""

from __future__ import annotations

__all__ = ["One2ManyTransformerAdapter"]

from abc import ABC
from typing import List, Tuple, TYPE_CHECKING

import torch
import torch.utils.data as tud

from .sample_batch import SampleBatchZ
from reinvent.models.model_factory.model_adapter import (
    ModelAdapter,
    SampledSequencesDTO,
    BatchLikelihoodDTO,
    BatchLikelihoodZDTO
)
from reinvent.models.transformer.core.dataset.paired_dataset import PairedDatasetWithZ
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum

from reinvent.models.model_factory.transformer_adapter import TransformerAdapter


if TYPE_CHECKING:
    pass


class One2ManyTransformerAdapter(TransformerAdapter):
    def likelihood(self, src, src_mask, trg, trg_mask, z) -> torch.Tensor:
        z_s_s = torch.cat([z_s_s[None] for z_t_s, z_s_s in z], 0)
        z_t_s = torch.cat([z_t_s[None] for z_t_s, z_s_s in z], 0)
        z = [z_t_s, z_s_s]
        return self.model.likelihood(src, src_mask, trg, trg_mask, z=z)

    def likelihood_smiles(
        self, sampled_sequence_list: List[SampledSequencesDTO]
    ) -> BatchLikelihoodZDTO:
        input = [dto.input for dto in sampled_sequence_list]
        output = [dto.output for dto in sampled_sequence_list]
        z_t_s = sampled_sequence_list.z_t_s
        z_s_s = sampled_sequence_list.z_s_s
        z = [z_t_s, z_s_s]
        
        dataset = PairedDatasetWithZ(input, output, enc_z=z, vocabulary=self.vocabulary, tokenizer=self.tokenizer)
        data_loader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=PairedDatasetWithZ.collate_fn
        )
       

        for batch in data_loader:
            likelihood, z = self.likelihood(
                batch.input, batch.input_mask, batch.output, batch.output_mask, z = batch.z
            )
            dto = BatchLikelihoodZDTO(batch, likelihood, z)
            return dto

    def sample(self, src, src_mask, decode_type=SamplingModesEnum.MULTINOMIAL) -> Tuple:
        # input SMILES, output SMILES, NLLs
        sampled = self.model.sample(src, src_mask, decode_type)
        return SampleBatchZ(*sampled)

    def set_beam_size(self, beam_size: int):
        self.model.set_beam_size(beam_size)

    def set_temperature(self, temperature: float = 1.0):
        self.model.set_temperature(temperature)
