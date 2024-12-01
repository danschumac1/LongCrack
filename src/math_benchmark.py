"""
Created on Mon 11 25 2024

@author: Dan Schumacher
TO RUN:
python ./src/math_benchmarking.py
"""
# IMPORTS
from utils.ClassAPI import api_config
import json
import pandas as pd

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using language models.")
    parser.add_argument("--output_path", type=str, required=True, help="Model name to use for generation.")
    return parser.parse_args()

def main():
    # LOAD THE DATA
    args = parse_args()
    subs_path = './data/cleaned.jsonl'
    df = pd.read_json(subs_path, lines=True)
    # to list
    system_prompt_config = json.load(open("resources/system_prompts.json"))
    system_prompt = system_prompt_config["math"]
  
    # API CONFIG
    client = api_config()

    # PROMPTING

    for i, row in df.iterrows():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": row["math_prompt"]
            }
            ],
            temperature=0.1,
            max_tokens=1024,
            top_p=0.1
        )
        resp = response.choices[0].message.content.strip()

        with open(args.output_path, 'a') as f:
            f.write(
                json.dumps(
                    {
                        "idx": i, 
                        "id": row["id"], 
                        "output": resp, 
                        "prompt": row["math_prompt"]}) + "\n")
        
if __name__ == "__main__":
    main()