'''
python ./src/generate.py \
    --model gpt-4o-mini \
    --all_questions false \
    --num_questions 1 \
    --dataset_path ./data/cleaned.jsonl \
    --step_size 1500 \
    --output_path ./test.jsonl \
    --prompt_key "math_prompt" \
    --system_prompt "long_math"
'''
import json
from utils.ClassAPI import DataProcessor, select_generator
from utils.logging import gen_logger
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using language models.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for generation.")
    parser.add_argument("--all_questions", type=str, default='false', choices=['true','false'], help="Use all malicious questions (set to true if present).")
    parser.add_argument("--num_questions", type=int, default=1, help="Number of malicious questions to use ('all' or an integer).")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to file containing malicious questions.")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for generating prompts.")
    parser.add_argument("--output_path", type=str, help="Output directory for generated text.")
    parser.add_argument("--prompt_key", type=str, default="question", help="Key to use for prompt generation")
    parser.add_argument("--system_prompt", type=str, default="math", choices=["math", "long_context", "long_math"],
                        help="System prompt to use for generation")
    return parser.parse_args()

def main():
    gen_logger(init=True)
    # PARSE ARGS
    args = parse_args()
    gen_logger("Arguments parsed successfully", "INFO")
    gen_logger(
        f"Model: {args.model}, All Questions: {args.all_questions}, Number of Questions: {args.num_questions}"
        f"Step Size: {args.step_size}, dataset_path: {args.dataset_path}, Output Path: {args.output_path}", "INFO")

    if args.all_questions == 'true':
        args.num_questions = 'all'
        
    gen_logger(f"Number of questions to use: {args.num_questions}", "INFO")

    # ENSURE OUTPUT DIRECTORY EXISTS
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # CLEAR OUTPUT FILE
    with open(args.output_path, 'w') as f:
        f.write("")
    gen_logger(f"Cleared existing output file at {args.output_path}", "INFO")       
    
    # DATA SETUP
    data_processor = DataProcessor()
    gen_logger("Loading benign questions...", "INFO")
    data_processor.load_benign_questions("data/benign_questions.jsonl")
    gen_logger("Loading malicious questions...", "INFO")
    data_processor.load_malicious_questions(
        path_to_jsonl=args.dataset_path, 
        num_questions=args.num_questions,
        prompt_key=args.prompt_key
        )
    data_processor.generate_list_of_prompts(args.step_size)
    gen_logger(f"Generated list of prompts with length: {len(data_processor.prompt_list)}", "INFO")

    # GENERATE TEXT
    gen_logger("Selecting and connecting to generator...", "INFO")
    generator = select_generator(args.model)
    generator.connect()
    gen_logger("Generator connected successfully", "INFO")

    system_prompt_config = json.load(open("resources/system_prompts.json"))
    system_prompt = system_prompt_config[args.system_prompt]
    
    # Iterate over prompts and generate completions
    for idx, (prompt, mal_q_id, insert_position, mal_question) in enumerate(data_processor.prompt_list):
        # Track the position of the malicious question in the benign list
        gen_logger(f"idx: {idx}\tmal_q_id: {mal_q_id}\tinsert_position: {insert_position}", "INFO")
        
        # Generate completion using the model
        try:
            output = generator.get_single_completion(
                model=args.model, 
                user_prompt=prompt, 
                malicious_uuid=data_processor.malicious_uuid,
                system_prompt=system_prompt
                )
        except Exception as e:
            gen_logger(f"Error for prompt #{idx}: {str(e)}", "ERROR")
            raise e

        # Build the JSON output structure        
        json_output = {
            "idx": idx + 1,
            "mal_q_id": mal_q_id,
            "insert_position": insert_position,
            "output": output,
            "mal_question": mal_question
        }

        # Write the output to a JSONL
        try:
            with open(args.output_path, 'a') as o_file:
                o_file.write(json.dumps(json_output) + "\n")
            gen_logger(f"Written output for prompt {idx} to {args.output_path}", "INFO")
        except IOError as e:
            gen_logger(f"Error writing output for prompt {idx} to file: {str(e)}", "ERROR")

    gen_logger("Run completed successfully", "INFO")

if __name__ == "__main__":
    main()