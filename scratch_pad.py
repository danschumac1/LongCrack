'''
python ./scratch_pad.py
'''

import pandas as pd 

def main():
    df_math = pd.read_json('./data/malq_math_prompt_pruned.jsonl', lines=True)    
    df_og = pd.read_json('./data/SimpleSafteyTest.jsonl', lines=True)
    # rename all columns in both to do lowercase
    df_math.columns = df_math.columns.str.lower()
    df_og.columns = df_og.columns.str.lower()

    # merge the two dataframes on df_math's question on df_og's question
    df_merged = pd.merge(df_math, df_og, left_on='question', right_on='question', how='inner')
    # rename "GPT Completion" to math_prompt

    df_merged.rename(columns={"gpt completion": "math_prompt"}, inplace=True)
    # save to a new jsonl file
    df_merged.to_json('./data/cleaned.jsonl', orient='records', lines=True)
    print("Done")
if __name__ == "__main__":
    main()