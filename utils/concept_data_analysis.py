import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


################################
#   Auxiliary functions
################################

def get_concept_by_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]


def get_statistics(filename, top_k_concepts):
    header = 0
    df = pd.read_csv(filename, header=header, sep="\t")

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

    most_frequent_concepts = [list(dict_concept_counter.values())[i] for i in sorted_concepts_desc]
    less_frequent_concepts = [list(dict_concept_counter.values())[i] for i in sorted_concepts_asc]

    list_of_concepts_per_image = []
    for i in range(max(list_cuis) + 1):
        value = int(len(np.where(list_cuis == i)[0]))
        list_of_concepts_per_image.append(value)

    return image_id, list_cuis, dict_concept_counter, most_frequent_concepts, less_frequent_concepts, list_of_concepts_per_image


def plot_bar_chart(list_cuis, list_of_concepts_per_image, title):
    # Plot chart
    bars = plt.bar(np.arange(0, max(list_cuis) + 1, 1), list_of_concepts_per_image)
    plt.bar_label(bars)
    plt.title(title)
    plt.xlabel('Number of Concepts')
    plt.ylabel('Number of Images')
    plt.xticks(np.arange(0, max(list_cuis) + 1, 1))
    plt.show()


if __name__ == "__main__":
    ################################
    # CSV files
    ################################

    file_name_training = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/Concept Detection/b47c4f80-9432-408c-b69a-956a3382a0da_ImageCLEFmedCaption_2022_concept_detection_train.csv"
    file_name_validation = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/Concept Detection/46bff9d5-95d4-4362-be98-ef59819ec3af_ImageCLEFmedCaption_2022_concept_detection_valid.csv"

    ################################
    # Training Data
    ################################

    top_k_concepts = 3
    image_id, list_cuis, dict_concept_counter, most_frequent_concepts, less_frequent_concepts, list_of_concepts_per_image = get_statistics(
        file_name_training, top_k_concepts=top_k_concepts)

    print("[INFO]: Statistics for Training Data")
    print(f"Number of training images: {len(image_id.keys())}")
    print(f"Average number of concepts per image: {list_cuis.mean()}")
    print(f"Minimum number of concepts per image: {min(list_cuis)}")
    print(f"Maximum number of concepts per image: {max(list_cuis)}")
    print(
        f"Less frequent concepts (top-{top_k_concepts}): {' '.join([get_concept_by_value(dict_concept_counter, c) for c in most_frequent_concepts])}")
    print(
        f"Most frequent concepts (top-{top_k_concepts}): {' '.join([get_concept_by_value(dict_concept_counter, c) for c in less_frequent_concepts])}")

    # Plot chart
    plot_bar_chart(list_cuis, list_of_concepts_per_image, title="Training Data")

    ################################
    # Validation Data
    ################################

    top_k_concepts = 3
    image_id, list_cuis, dict_concept_counter, most_frequent_concepts, less_frequent_concepts, list_of_concepts_per_image = get_statistics(
        file_name_validation, top_k_concepts=top_k_concepts)

    print("[INFO]: Statistics for Validation Data")
    print(f"Number of validation images: {len(image_id.keys())}")
    print(f"Average number of concepts per image: {list_cuis.mean()}")
    print(f"Minimum number of concepts per image: {min(list_cuis)}")
    print(f"Maximum number of concepts per image: {max(list_cuis)}")
    print(
        f"Less frequent concepts (top-{top_k_concepts}): {' '.join([get_concept_by_value(dict_concept_counter, c) for c in most_frequent_concepts])}")
    print(
        f"Most frequent concepts (top-{top_k_concepts}): {' '.join([get_concept_by_value(dict_concept_counter, c) for c in less_frequent_concepts])}")

    # Plot chart
    plot_bar_chart(list_cuis, list_of_concepts_per_image, title="Validation Data")
