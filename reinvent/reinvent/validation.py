"""Global Config Validation"""

from typing import Optional

from pydantic import BaseModel, ConfigDict


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces = ())


class ReinventConfig(BaseModel):
    run_type: str
    device: str = "cpu"
    tb_logdir: Optional[str] = None
    json_out_config: Optional[str] = None
