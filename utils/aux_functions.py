# Imports
import os
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
    concept_dict_name_to_desc = dict()

    for index, row in concepts_df.iterrows():
        concept_dict_name_to_idx[row["concept"]] = index
        concept_dict_idx_to_name[index] = row["concept"]
        concept_dict_name_to_desc[row["concept"]] = row["concept_name"]

    # print(concept_dict_name_to_idx)
    # print(concept_dict_idx_to_name)
    # print(concept_dict_name_to_desc)

    return concept_dict_name_to_idx, concept_dict_idx_to_name, concept_dict_name_to_desc



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



# Function: Compute pos_weights
def compute_pos_weights(n_concepts):
    if n_concepts == 100:
        csv_path_dataset = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/ImageClefMedical/data/dataset_resized/new_train_subset_top100.csv" #"../data/dataset_resized/new_train_subset_top100.csv"
        csv_path_concepts = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/ImageClefMedical/data/dataset_resized/new_top100_concepts.csv"
    else:
        csv_path_dataset = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/ImageClefMedical/data/dataset_resized/concept_detection_train.csv"
        csv_path_concepts = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/ImageClefMedical/data/dataset_resized/concepts.csv"

    # Get DatFrame from train dataset file
    df_dataset = pd.read_csv(csv_path_dataset, header=0, sep="\t")

    # Number of examples
    N = len(df_dataset)

    # Get DataFrame from concept's file
    df_all_concepts = pd.read_csv(csv_path_concepts, header=0, sep="\t")
    all_concepts = df_all_concepts["concept"]

    # Convert concepts to a matrix of 1s and 0s denoting the presence/absence of each concept in the image
    dict_concept = dict()
    for idx, c in enumerate(all_concepts):
        dict_concept[c] = idx

    matrix = np.zeros((len(df_dataset["ID"]), len(all_concepts)))
    for i in range(len(df_dataset["ID"])):
        dict_concepts_per_image = df_dataset["cuis"][i].split(";")
        for c in dict_concepts_per_image:
            matrix[i][dict_concept[c]] = 1

    # pos_weights = neg / pos
    pos_count = np.count_nonzero(matrix, axis=0)
    neg_count = N - pos_count
    np.testing.assert_array_equal(np.ones_like(pos_count)*N, np.sum((neg_count, pos_count), axis=0))
    pos_weights = neg_count / pos_count

    return pos_weights



# Function: Compute class_weights
def get_class_weights(n_concepts):
    if n_concepts == 100:
        csv_path_dataset = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/ImageClefMedical/data/dataset_resized/new_train_subset_top100.csv" #"../data/dataset_resized/new_train_subset_top100.csv"
        csv_path_concepts = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/ImageClefMedical/data/dataset_resized/new_top100_concepts.csv"
    else:
        csv_path_dataset = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/ImageClefMedical/data/dataset_resized/concept_detection_train.csv"
        csv_path_concepts = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/ImageClefMedical/data/dataset_resized/concepts.csv"

    # Get DatFrame from train dataset file
    df_dataset = pd.read_csv(csv_path_dataset, header=0, sep="\t")

    # Number of examples
    N = len(df_dataset)

    # Get DataFrame from concept's file
    df_all_concepts = pd.read_csv(csv_path_concepts, header=0, sep="\t")
    all_concepts = df_all_concepts["concept"]

    # Convert concepts to a matrix of 1s and 0s denoting the presence/absence of each concept in the image
    dict_concept = dict()
    for idx, c in enumerate(all_concepts):
        dict_concept[c] = idx

    matrix = np.zeros((len(df_dataset["ID"]), len(all_concepts)))
    for i in range(len(df_dataset["ID"])):
        dict_concepts_per_image = df_dataset["cuis"][i].split(";")
        for c in dict_concepts_per_image:
            matrix[i][dict_concept[c]] = 1

    activated_concepts = np.count_nonzero(matrix, axis=0)
    class_weights = N / (n_concepts * activated_concepts)

    return class_weights



# Function: Map concepts to semantic types
def map_concepts_to_semantic(concepts_df, semantic_types_df, column="concept"):
    
    # Join the two concepts on "concept"
    new_df = concepts_df.copy().join(other=semantic_types_df.copy(), on=column, how='left')

    # Drop NaNs
    new_df = new_df.copy().dropna(axis=0)


    return new_df
