# System Imports
import os
import sys

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


# Get top-k most frequent concepts
_, _, _, most_frequent_concepts, _, _ = get_statistics(filename=os.path.join(DATA_DIR, CONCEPTS_TRAIN), top_k_concepts=TOP_K_CONCEPTS)
print(most_frequent_concepts)
