#!/bin/bash
# -----------------------------------------------------------
# To Run:   ./bin/lcm_generation.sh 
#           nohup ./bin/lcm_generation.sh &
# -----------------------------------------------------------

# CHANGE THESE
model="gpt-4o-mini"
step_size=500
num_questions=66
all_questions="true"
dataset_path="./data/cleaned.jsonl"
prompt_key="math_prompt"
system_prompt="long_math"


# THESE STAY THE SAME
output_path="./data/generations/$system_prompt.jsonl"

python ./src/generate.py \
    --model=${model} \
    --num_question=${num_questions} \
    --step_size=${step_size} \
    --all_questions=${all_questions} \
    --dataset_path=${dataset_path} \
    --prompt_key=${prompt_key} \
    --system_prompt=${system_prompt} \
    --output_path=${output_path}

