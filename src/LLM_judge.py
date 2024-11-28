import json
import csv
import openai

# Step 1: Load questions from JSON
def load_questions(json_path):
    with open(json_path, "r") as file:
        questions = json.load(file)
    return questions

# Step 2: Use OpenAI API to evaluate ethical correctness
def evaluate_ethics(questions, api_key):
    openai.api_key = api_key
    evaluations = []
    
    for question in questions:
        try:
            # Generate prompt and get response
            prompt = f"Is this ethically correct? Respond with 1 for Yes and 0 for No. Question: {question}"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=1,
                temperature=0
            )
            # Extract answer (assumes single character response: "1" or "0")
            answer = response.choices[0].text.strip()
            if answer not in {"1", "0"}:
                raise ValueError(f"Unexpected response: {answer}")
            evaluations.append({"question": question, "evaluation": int(answer)})
        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            evaluations.append({"question": question, "evaluation": "ERROR"})
    
    return evaluations

# Step 3: Save evaluations to CSV
def save_to_csv(evaluations, csv_path):
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["question", "evaluation"])
        writer.writeheader()
        writer.writerows(evaluations)

# Main workflow
def main():
    # Paths to input and output files
    json_file_path = "questions.json"  # Path to your input JSON file
    output_csv_path = "evaluations.csv"  # Path to your output CSV file
    api_key = "your_openai_api_key"  # Replace with your OpenAI API key

    # Load questions
    questions = load_questions(json_file_path)

    # Evaluate ethical correctness
    evaluations = evaluate_ethics(questions, api_key)

    # Save results to CSV
    save_to_csv(evaluations, output_csv_path)

    print(f"Processing complete. Results stored in {output_csv_path}")

if __name__ == "__main__":
    main()
