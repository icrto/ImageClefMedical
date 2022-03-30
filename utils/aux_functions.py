# Imports
import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Function: Get concepts by value
def get_concepts_dicts(data_dir, concepts_csv):
    
    """
    data_dir (str): path to the database
    concepts_csv (str): name of the .CSV file that contains the concepts
    """

    # Open concepts
    concepts_df = pd.read_csv(os.path.join(data_dir, concepts_csv), sep="\t")

    # Create index dict to map concepts later  
    concept_dict_name_to_idx = dict()
    concept_dict_idx_to_name = dict()

    for index, row in concepts_df.iterrows():
        concept_dict_name_to_idx[row["concept"]] = index
        concept_dict_idx_to_name[index] = row["concept"]

    # print(concept_dict)
    

    return concept_dict_name_to_idx, concept_dict_idx_to_name



# Function: Get statistics
def get_statistics(filename, top_k_concepts):

    # Load .CSV
    df = pd.read_csv(filename, sep="\t")

    image_id = df["ID"]
    cuis = df["cuis"]

    dict_concepts_per_image = dict()
    for i, im_id in enumerate(image_id.keys()):
        dict_concepts_per_image[im_id] = cuis[i].split(";")

    list_cuis = []
    dict_concept_counter = dict()
    for c in cuis:
        split = c.split(";")
        list_cuis.append(len(split))
        for s in split:
            if s not in dict_concept_counter:
                dict_concept_counter[s] = 1
            else:
                dict_concept_counter[s] += 1

    list_cuis = np.asarray(list_cuis)

    # Get top-k most/less frequent concepts
    k = top_k_concepts
    sorted_concepts_desc = np.argsort(list(dict_concept_counter.values()))[::-1][:k]
    sorted_concepts_asc = np.argsort(list(dict_concept_counter.values()))[:k]

    most_frequent_concepts = [list(dict_concept_counter.keys())[i] for i in sorted_concepts_desc]
    less_frequent_concepts = [list(dict_concept_counter.keys())[i] for i in sorted_concepts_asc]

    list_of_concepts_per_image = []
    for i in range(max(list_cuis) + 1):
        value = int(len(np.where(list_cuis == i)[0]))
        list_of_concepts_per_image.append(value)

    return image_id, list_cuis, dict_concept_counter, most_frequent_concepts, less_frequent_concepts, list_of_concepts_per_image



# Function: Plot bar chart with statistics
def plot_bar_chart(list_cuis, list_of_concepts_per_image, title):
    # Plot chart
    bars = plt.bar(np.arange(0, max(list_cuis) + 1, 1), list_of_concepts_per_image)
    plt.bar_label(bars)
    plt.title(title)
    plt.xlabel('Number of Concepts')
    plt.ylabel('Number of Images')
    plt.xticks(np.arange(0, max(list_cuis) + 1, 1))
    plt.show()
