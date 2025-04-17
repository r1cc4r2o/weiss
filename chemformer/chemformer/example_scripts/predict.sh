#!/bin/bash

python -m molbart.predict \
  --reactants_path uspto_50_test.txt \
  --products_path uspto_50_p0_5_out.pickle \
  --model_path model_path \
  --batch_size 64 \
  --n_beams 10