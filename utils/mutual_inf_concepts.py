# Imports
import os
import pandas as pd
import numpy as np

# Sklearn Imports
from sklearn.feature_selection import mutual_info_classif



# Directories and Files
data_dir = "data/dataset_resized"
concepts_csv = "concepts.csv"
concepts_train = "concept_detection_train.csv"


# Open concepts
concepts_df = pd.read_csv(os.path.join(data_dir, concepts_csv), sep="\t")

# Create index dict to map concepts later  
concept_dict = dict()
for index, row in concepts_df.iterrows():
    concept_dict[row["concept"]] = index

# print(concept_dict)



# Open train concepts
concepts_train_df = pd.read_csv(os.path.join(data_dir, concepts_train), sep="\t")

# Create square-matrix to store mutual information values among concepts
# print(len(concepts_train_df))

# Create an NumPy array with the dataset
data_arr = np.zeros(shape=(len(concepts_train_df), len(concept_dict)))
# print(data_arr.shape)


# Iterate through concepts_train
for index, row in concepts_train_df.iterrows():
    
    # Parse concepts
    concepts = row["cuis"].split(';')
    # print(concepts)

    # Populate data array with ones where we have concepts
    for c in concepts:
        data_arr[index, concept_dict[c]] = 1



# Mutual Information Array
mi_array = np.zeros(shape=(len(concept_dict), len(concept_dict)))


# Iterate through all the samples
for idx_c in range(len(concept_dict)):
    
    # Define features array
    features = data_arr.copy()

    # Define target array
    target = data_arr.copy()[:, idx_c]


    # Compute Mutual Information
    mi = mutual_info_classif(X=features, y=target, random_state=42)


    # Populate Mutual Information Array
    mi_array[idx_c, :] = mi
    print(mi_array[idx_c, :])


print("Finished")
