#!/bin/bash
# nohup ./bin/lcm_judge.sh &

python ./src/llm_as_judge.py \
    --input_path ./data/generations/long_math.jsonl \
    --output_path ./data/results/long_math_results.jsonl