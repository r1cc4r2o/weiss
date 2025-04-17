#!/bin/bash

export HYDRA_FULL_ERROR=1
python -m molbart.fine_tune 'hydra.searchpath=[./data/current/hydra_template]' experiment=train.yaml
