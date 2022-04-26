# Imports
import os
import sys
import argparse
import numpy as np
import pandas as pd


# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


# Custom Imports
from aux_functions import map_concepts_to_semantic



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data/csv", help="Directory of the data set.")

# Concepts .CSV 
parser.add_argument('--concepts_csv', type=str, default="concepts.csv", help="Name of the .CSV concepts file.")

# Semantic Types .CSV
parser.add_argument('--semantic_types_csv', type=str, default="semantic_types.csv", help="Name of the .CSV semantic types file.")


# Parse the arguments
args = parser.parse_args()



# Directories and Files
DATA_DIR = args.data_dir
CONCEPTS_CSV = args.concepts_csv
SEMANTIC_TYPES_CSV = args.semantic_types_csv



# Load .CSV files
concepts_df = pd.read_csv(os.path.join(DATA_DIR, CONCEPTS_CSV), sep="\t")
semantic_types_df = pd.read_csv(os.path.join(DATA_DIR, SEMANTIC_TYPES_CSV), sep="\t")


# Get processed DataFrame
processed_df = map_concepts_to_semantic(concepts_df=concepts_df, semantic_types_df=semantic_types_df)

# Convert the DataFrame to a .CSV file
processed_df.to_csv(os.path.join(DATA_DIR, f"{CONCEPTS_CSV.split('.')[0]}_sem.csv"), sep="\t", index=False)


# Check if this is working
processed_df = pd.read_csv(os.path.join(DATA_DIR, f"{CONCEPTS_CSV.split('.')[0]}_sem.csv"), sep="\t")
print(processed_df)


print("Finished")
