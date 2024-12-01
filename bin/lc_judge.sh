#!/bin/bash
# nohup ./bin/lc_judge.sh &

python ./src/llm_as_judge.py \
    --input_path ./data/generations/long_context.jsonl \
    --output_path ./data/results/long_context_results.jsonl