#!/bin/bash

python -m molbart.inference_score \
  --data_path ../data/uspto_sep.pickle \
  --model_path model_path \
  --dataset_type uspto_sep \
  --task forward_prediction \
  --model_type bart \
  --batch_size 64 \
  --n_beams 10