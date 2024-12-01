'''
Run the results processing script.
'''

import json
from typing import List, Optional
import pandas as pd
from scipy import stats  # Importing scipy for statistical analysis
import seaborn as sns
import matplotlib.pyplot as plt
import os  # For creating directories


def safe_string_to_int(s: str) -> Optional[int]:
    """
    Safely convert a string to an integer.
    If conversion fails, returns None and logs the issue.
    """
    try:
        return int(s)
    except ValueError:
        print(f"CRAPP!!! Could not convert: {s}")
        return None


def accuracy(zero_ones: List[int]) -> float:
    """
    Calculate accuracy given a list of binary values (0s and 1s).
    Accuracy is the sum of the list divided by its length.
    """
    if not zero_ones:
        print("Empty list provided for accuracy calculation.")
        return 0.0
    return sum(zero_ones) / len(zero_ones)


def process_and_analyze(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Process the DataFrame, print the overall accuracy,
    and calculate accuracy by 'insert_position' if the column exists.
    Returns the processed DataFrame.
    """
    # Convert responses to integers safely
    df["response"] = df["response"].apply(safe_string_to_int)

    # Drop rows with invalid responses
    df = df.dropna(subset=["response"])

    # Print overall accuracy
    print(f"\n{name} Overall accuracy: {accuracy(df['response'].tolist())}")

    # Analyze by insert position if the column exists
    if "insert_position" in df.columns:
        by_pos = df.groupby("insert_position")["response"]
        for pos, responses in by_pos:
            print(f"Accuracy for position {pos}: {accuracy(responses.tolist())}")
    
    return df


def perform_statistical_test(df: pd.DataFrame):
    """
    Perform a one-way ANOVA test to determine if differences in
    means across 'insert_position' groups are significant.
    """
    if "insert_position" not in df.columns:
        print("No 'insert_position' column found for statistical testing.")
        return

    # Group responses by insert position
    groups = [group["response"].tolist() for _, group in df.groupby("insert_position")]

    # Perform ANOVA
    try:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\nANOVA Results:")
        print(f"F-statistic: {f_stat}, p-value: {p_value}")
        if p_value < 0.05:
            print("Result: Significant difference detected between groups.")
        else:
            print("Result: No significant difference detected between groups.")
    except Exception as e:
        print(f"Failed to perform statistical test: {e}")


def save_bar_chart(df: pd.DataFrame, output_dir: str, filename: str):
    """
    Save a bar chart visualizing accuracy by insert position for the given DataFrame.
    """
    if "insert_position" in df.columns:
        # Calculate mean accuracy by position
        accuracy_by_position = df.groupby("insert_position")["response"].mean().reset_index()

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x="insert_position", y="response", data=accuracy_by_position, palette="viridis")
        plt.title("Accuracy by Insert Position (Long Math Results)", fontsize=14)
        plt.xlabel("Insert Position", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Bar chart saved to {output_path}")


def main():
    """
    Main function to load data, process it, analyze results, and perform statistical tests.
    """
    # Load data
    file_paths = {
        "math_results": "./data/results/math_results.jsonl",
        "long_context_results": "./data/results/long_context_results.jsonl",
        "long_math_results": "./data/results/long_math_results.jsonl",
    }

    dataframes = {}
    for key, path in file_paths.items():
        try:
            dataframes[key] = pd.read_json(path, lines=True)
        except Exception as e:
            print(f"Failed to load {key} from {path}: {e}")
            return

    # Process and analyze each DataFrame
    processed_dataframes = {}
    for name, df in dataframes.items():
        print(f"\nAnalyzing {name}...")
        processed_dataframes[name] = process_and_analyze(df, name)
        
        # Perform statistical tests on 'long_math_results' DataFrame only
        if name == "long_math_results":
            print("\nLONG MATH ANOVA...")
            perform_statistical_test(processed_dataframes[name])

            # Save bar chart for long math results
            save_bar_chart(processed_dataframes[name], output_dir="./figures", filename="long_math_accuracy.png")

    # Perform a t-test on the overall accuracy of math_results vs long_math_results
    print("\nT-TEST...")
    math_results = processed_dataframes["math_results"]
    long_math_results = processed_dataframes["long_math_results"]

    t_stat, p_value = stats.ttest_ind(
        math_results["response"].dropna(), long_math_results["response"].dropna()
    )

    print(f"Math results accuracy: {accuracy(math_results['response'].tolist())}")
    print(f"Long math results accuracy: {accuracy(long_math_results['response'].tolist())}")
    print(f"T-statistic: {t_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Result: Significant difference detected between math and long math results.")
    else:
        print("Result: No significant difference detected between math and long math results.")


if __name__ == '__main__':
    main()
