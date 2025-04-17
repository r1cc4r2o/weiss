"""Config Validation"""

from typing import List, Optional

from pydantic import Field

from reinvent.validation import GlobalConfig


class SectionParameters(GlobalConfig):
    prior_file: str
    agent_file: str
    summary_csv_prefix: str = "summary"
    use_checkpoint: bool = False
    purge_memories: bool = True
    smiles_file: Optional[str] = None  # not Reinvent
    sample_strategy: Optional[str] = None  # Transformer
    distance_threshold: int = 99999  # Transformer
    batch_size: int = 100
    randomize_smiles: bool = True
    unique_sequences: bool = False
    temperature: Optional[float] = 0.1


class SectionLearningStrategy(GlobalConfig):
    type: str = "dap"
    sigma: int = 128
    rate: float = 0.0001


class SectionDiversityFilter(GlobalConfig):
    type: str
    bucket_size: int = 25
    minscore: float = 0.4
    minsimilarity: float = 0.4
    penalty_multiplier: float = 0.5


class SectionInception(GlobalConfig):
    smiles_file: Optional[str] = None
    memory_size: int = 50
    sample_size: int = 10
    deduplicate: bool = True


class SectionStage(GlobalConfig):
    max_steps: int
    max_score: float
    chkpt_file: Optional[str] = None
    termination: str = "simple"
    min_steps: int = 50
    scoring: dict = Field(default_factory=dict)  # validate in Scorer
    diversity_filter: Optional[SectionDiversityFilter] = None


# FIXME: may only need this once
class SectionResponder(GlobalConfig):
    endpoint: str
    frequency: int = 1


class RLConfig(GlobalConfig):
    parameters: SectionParameters
    stage: List[SectionStage]
    learning_strategy: Optional[SectionLearningStrategy] = None
    diversity_filter: Optional[SectionDiversityFilter] = None
    inception: Optional[SectionInception] = None
    responder: Optional[SectionResponder] = None
