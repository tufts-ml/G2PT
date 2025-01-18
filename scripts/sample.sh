#!/bin/bash

python sample.py  \
    --num_sample 25000 \
    --batch_size 512 \
    --seed 1337 \
    --out_dir results/guacamol-base-deg \
    --tokenizer_path tokenizers/guacamol