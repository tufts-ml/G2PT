#!/bin/bash

 torchrun --standalone --nproc_per_node=4 train.py \
   configs/networks/small.py \
   configs/datasets/moses.py \
   configs/default.py
   
