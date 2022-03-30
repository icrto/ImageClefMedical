# System Imports
import os
import sys
import tqdm
import numpy as np
import pandas as pd

# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


# Custom Imports
from utils.aux_functions import get_concepts_dicts, get_statistics



# Directories and Files
DATA_DIR = "data/dataset_resized"
CONCEPTS_CSV = "concepts.csv"
CONCEPTS_TRAIN = "concept_detection_train.csv"
CONCEPTS_VAL = "concept_detection_valid.csv"
TOP_K_CONCEPTS = 100



# Generate concepts dictionaries
concept_dict_name_to_idx, concept_dict_idx_to_name = get_concepts_dicts(data_dir=DATA_DIR, concepts_csv=CONCEPTS_CSV)


# Get top-k most frequent concepts (in training set)
_, _, _, most_frequent_concepts, _, _ = get_statistics(filename=os.path.join(DATA_DIR, CONCEPTS_TRAIN), top_k_concepts=TOP_K_CONCEPTS)
# print(most_frequent_concepts)


# Open train subset
concepts_train_df = pd.read_csv(os.path.join(DATA_DIR, CONCEPTS_TRAIN), sep="\t")

# Iterate through train subset and generate a new subset with these concepts
new_train_subset = list()

for index, row in tqdm.tqdm(concepts_train_df.iterrows()):
    
    # Parse image and concepts
    image = row["ID"]
    concepts = row["cuis"].split(';')
    # print(concepts)

    # Populate data array with ones where we have concepts
    for c in concepts:
        if c in most_frequent_concepts:
            new_train_subset.append([image, concepts])
            break
    
print(len(new_train_subset))



# Open validation subset
concepts_val_df = pd.read_csv(os.path.join(DATA_DIR, CONCEPTS_VAL), sep="\t")

# Iterate through train subset and generate a new subset with these concepts
new_val_subset = list()

for index, row in tqdm.tqdm(concepts_val_df.iterrows()):
    
    # Parse image and concepts
    image = row["ID"]
    concepts = row["cuis"].split(';')
    # print(concepts)

    # Populate data array with ones where we have concepts
    for c in concepts:
        if c in most_frequent_concepts:
            new_val_subset.append([image, concepts])
            break
    
print(len(new_val_subset))


print("Finished.")
