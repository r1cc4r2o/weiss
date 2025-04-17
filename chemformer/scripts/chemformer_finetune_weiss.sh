#!/bin/bash

export HYDRA_FULL_ERROR=1
python -m molbart.fine_tune 'hydra.searchpath=[./data/hydra_template]' experiment=fine_tune_sp_wae.yaml
