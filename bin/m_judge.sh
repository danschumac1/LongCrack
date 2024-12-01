#!/bin/bash
# nohup ./bin/m_judge.sh &

python ./src/llm_as_judge.py \
    --input_path ./data/generations/math.jsonl \
    --output_path ./data/results/math_results.jsonl