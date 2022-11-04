# System Imports
import os
import sys
import tqdm
import argparse
import pandas as pd


# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


# Custom Imports
from aux_functions import get_concepts_dicts, get_statistics


# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str,
                    default="dataset", help="Directory of the data set.")

# Concepts .CSV file
parser.add_argument('--concepts_csv', type=str,
                    default="concepts.csv", help="Concepts .CSV file.")

# Concepts Train .CSV file
parser.add_argument('--concepts_train', type=str,
                    default="concept_detection_train.csv", help="Concepts Train .CSV file.")

# Concepts Val .CSV file
parser.add_argument('--concepts_val', type=str,
                    default="concept_detection_valid.csv", help="Concepts Val .CSV file.")

# Captions Train .CSV file
parser.add_argument('--captions_train', type=str,
                    default="caption_prediction_train.csv", help="Captions Train .CSV file.")

# Captions Val .CSV file
parser.add_argument('--captions_val', type=str,
                    default="caption_detection_valid.csv", help="Captions Val .CSV file.")

# Top-K concepts
parser.add_argument('--top_k', type=int, default=100,
                    help="The top-K concepts you want to extract.")

# Column name
parser.add_argument('--column_name', type=str, default="concept_name",
                    help="The column name to map the concepts.")

# Semantic Types
parser.add_argument('--semantic_types', action="store_true",
                    help="Activate if you are using semantic types.")


# Parse the arguments
args = parser.parse_args()


# Directories and Files
DATA_DIR = args.data_dir
CONCEPTS_CSV = args.concepts_csv
CONCEPTS_TRAIN = args.concepts_train
CONCEPTS_VAL = args.concepts_val
CAPTIONS_TRAIN = args.captions_train
CAPTIONS_VAL = args.captions_val
TOP_K_CONCEPTS = args.top_k
COLUMN_NAME = args.column_name
SEMANTIC_TYPES = args.semantic_types


# Generate concepts dictionaries
concept_dict_name_to_idx, concept_dict_idx_to_name, concept_dict_name_to_desc = get_concepts_dicts(
    data_dir=DATA_DIR, concepts_csv=CONCEPTS_CSV, column=COLUMN_NAME)

# Get top-k most frequent concepts (in training set)
_, _, _, most_frequent_concepts, _, _ = get_statistics(
    filename=os.path.join(DATA_DIR, CONCEPTS_TRAIN), top_k_concepts=TOP_K_CONCEPTS)

# Create a list with descriptions of the most frequent concepts
most_frequent_concepts_desc = [
    concept_dict_name_to_desc.get(c) for c in most_frequent_concepts]

# Create new .CSV of concepts
# Create a dictionary to obtain DataFrame later
new_csv_concepts = {'concept': most_frequent_concepts, 'concept_name': most_frequent_concepts_desc}

# Save this into .CSV
new_csv_concepts_df = pd.DataFrame(data=new_csv_concepts)
if SEMANTIC_TYPES:
    new_csv_concepts_df.to_csv(os.path.join(
        DATA_DIR, f"concepts_top{TOP_K_CONCEPTS}_sem.csv"), sep="\t", index=False)

else:
    new_csv_concepts_df.to_csv(os.path.join(
        DATA_DIR, f"concepts_top{TOP_K_CONCEPTS}.csv"), sep="\t", index=False)

for fconcepts, fcaptions in zip([CONCEPTS_TRAIN, CONCEPTS_VAL], [CAPTIONS_TRAIN, CAPTIONS_VAL]):
    # Open subset
    concepts_df = pd.read_csv(
        os.path.join(DATA_DIR, fconcepts), sep="\t")

    # Iterate through set and generate a new subset with these concepts
    new_images = list()
    new_concepts = list()

    for index, row in tqdm.tqdm(concepts_df.iterrows()):

        # Parse image and concepts
        image = row["ID"]
        concepts_list = row["cuis"].split(';')

        # Add the valid concepts
        new_concepts_iter = ""
        for c in concepts_list:
            if c in most_frequent_concepts:
                new_concepts_iter += f"{c};"

        if len(new_concepts_iter) > 0:
            new_concepts_iter = new_concepts_iter[:-1]
            new_images.append(image)
            new_concepts.append(new_concepts_iter)


    # Create a dictionary to obtain DataFrame later
    new_subset = {'ID': new_images, 'cuis': new_concepts}

    # Save this into .CSV
    new_subset_df = pd.DataFrame(data=new_subset)
    if SEMANTIC_TYPES:
        new_subset_df.to_csv(os.path.join(
            DATA_DIR, fconcepts.replace('.csv', '_top{TOP_K_CONCEPTS}_sem.csv')), sep="\t", index=False)

    else:
        new_subset_df.to_csv(os.path.join(
            DATA_DIR, fconcepts.replace('.csv', '_top{TOP_K_CONCEPTS}.csv')), sep="\t", index=False)

    
    # Filter caption csv to contain only valid images
    caption_df = pd.read_csv(fcaptions, sep="\t")
    new_caption_df = caption_df.loc[caption_df['ID'].isin(new_images)]
    print(f"Had {len(caption_df)} captions now have {len(new_caption_df)} captions")

    new_caption_df.to_csv(os.path.join(
            DATA_DIR, fcaptions.replace('.csv', '_top{TOP_K_CONCEPTS}.csv')), index=None, sep='\t')

print("Finished.")
