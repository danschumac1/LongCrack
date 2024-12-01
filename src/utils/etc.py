import json
import os
from dotenv import load_dotenv
import tiktoken

def token_counter(text: str, model: str = "gpt-4o-mini"):
    # Load the tokenizer for the model
    encoding = tiktoken.encoding_for_model(model)

    # Tokenize the input
    tokenized_input = encoding.encode(text)

    # Count the number of tokens (list length)
    token_count = len(tokenized_input)

    return token_count

def load_json_list(json_path: str) -> list:
    """
    Load a list of questions from a JSON file.
    Args:
        json_path (str): Path to the JSON file.
    Returns:
        list: List of questions loaded from the JSON file.
    """
    with open(json_path, "r") as file:
        return json.load(file)
