# Imports
import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm

# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Csvs directory
parser.add_argument('--preds_dir', type=str, required=True,
                    help="Directory of the csv files with the predictions of each of the 9 models.")

# Parse the arguments
args = parser.parse_args()

csv_files = [c for c in os.listdir(args.preds_dir) if not c.startswith('.') and c.endswith('.csv') and not 'agg' in c]

assert len(csv_files) == 9, f"9 csv files are needed, but {len(csv_files)} were provided."

# Pre-create image lists
eval_data = dict()

# Go through all the .CSV files
print('Aggregating...')
for csv in tqdm(csv_files):

    # Read .CSV
    df = pd.read_csv(os.path.join(args.preds_dir, csv), sep="|", header=None)

    # Create image list
    for _, row in df.iterrows():
        if row[0] not in eval_data.keys():
            eval_data[row[0]] = list()

    # Append concepts (if different from 'None')
    for index, row in df.iterrows():

        eval_data[row[0]] += str(row[1]).split(';')

        for i, c in enumerate(eval_data[row[0]]):
            if c in ("None", "nan"):
                eval_data[row[0]].pop(i)

        # Remove duplicates if needed (we don't know why this happens)
        eval_data[row[0]] = list(dict.fromkeys(eval_data[row[0]]))

# Process concept lists
for key, value in eval_data.items():
    # Add the valid concepts
    predicted_concepts = ""
    for c in value:
        predicted_concepts += f"{c};"

    eval_data[key] = predicted_concepts[:-1]

# Convert this data into a DataFrame
df_dict = dict()
df_dict["ID"] = list()
df_dict["cuis"] = list()
for key, value in eval_data.items():
    df_dict["ID"].append(key)
    df_dict["cuis"].append(value)

evaluation_df = pd.DataFrame(data=df_dict)
fname = os.path.join(args.preds_dir, "preds_agg.csv")
evaluation_df.to_csv(fname, sep="|", index=False, header=False)
print(f'Saved results to: {fname}')
