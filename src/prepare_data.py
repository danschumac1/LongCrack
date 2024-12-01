"""
Created on 10/15/2024

@author: Dan
python ./src/prepare_data.py
"""
from utils.etc import token_counter
import pandas as pd
import json
import uuid
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default='./data/original/list_of_questions.json')
    parser.add_argument("--output_file_path", type=str, default='./data/benign_questions.jsonl')
    parser.add_argument("--max_tokens", type=int, default=128000, help="Maximum token length for output file")
    return parser.parse_args()

def write_to_jsonl(file_path, data_generator):
    """Utility to write a generator to a JSONL file."""
    with open(file_path, 'w') as file:
        for record in data_generator:
            file.write(json.dumps(record) + '\n')

def main():
    args = parse_args()
    
    # Load input data
    with open(args.input_file_path, 'r') as i_file:
        questions = json.load(i_file)

    df_prompt = pd.read_json('./data/cleaned.jsonl', lines=True)
    longest_math_prompt = df_prompt['math_prompt'].apply(lambda x: len(x)).max()

    # Generate UUIDs and prepare for output
    output_list = [{"question": q, "uuid": str(uuid.uuid4())} for q in questions]

    # Iteratively fill data to respect token limit
    max_tokens = args.max_tokens - longest_math_prompt - 100  # Dynamic buffer
    current_tokens = 0

    def data_generator():
        nonlocal current_tokens
        index = 0
        while current_tokens < max_tokens:
            record = output_list[index % len(output_list)]
            record_text = json.dumps(record)
            record_tokens = token_counter(record_text)

            if current_tokens + record_tokens > max_tokens:
                break
            
            current_tokens += record_tokens
            yield record
            
            index += 1

    # Write the generated data to file
    write_to_jsonl(args.output_file_path, data_generator())

if __name__ == "__main__":
    main()