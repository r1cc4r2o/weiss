#!/bin/bash

export HYDRA_FULL_ERROR=1
python -m molbart.inference_score 'hydra.searchpath=[./data/current/hydra_template]' experiment=inference_score_bart.yaml
