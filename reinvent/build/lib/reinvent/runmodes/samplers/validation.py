"""Config Validation"""

from typing import Optional

from reinvent.validation import GlobalConfig


class SectionParameters(GlobalConfig):
    model_file: str
    num_smiles: int
    smiles_file: Optional[str] = None
    target_smiles_path: str = ""
    sample_strategy: Optional[str] = "multinomial"  # Mol2Nol
    output_file: str = "samples.csv"
    unique_molecules: bool = True
    randomize_smiles: bool = True
    temperature: Optional[float] = 1.0  # Mol2Mol


class SamplingConfig(GlobalConfig):
    parameters: SectionParameters
