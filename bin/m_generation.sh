#!/bin/bash
# -----------------------------------------------------------
# To Run:   ./bin/m_generation.sh 
#           nohup ./bin/m_generation.sh &
# -----------------------------------------------------------
# THESE STAY THE SAME
output_path="./data/generations/math.jsonl"

python ./src/math_benchmark.py \
    --output_path=${output_path}
