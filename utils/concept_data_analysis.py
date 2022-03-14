import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name_training = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/Concept Detection/b47c4f80-9432-408c-b69a-956a3382a0da_ImageCLEFmedCaption_2022_concept_detection_train.csv"
file_name_validation = "/home/cristianopatricio/Desktop/PhD/ImageCLEFmedCaption 2022/Concept Detection/46bff9d5-95d4-4362-be98-ef59819ec3af_ImageCLEFmedCaption_2022_concept_detection_valid.csv"

################################
#   Training Data
################################
header = 0
df = pd.read_csv(file_name_training, header=header, sep="\t")

cuis = df["cuis"]

list_cuis = []
for c in cuis:
    split = c.split(";")
    list_cuis.append(len(split))
list_cuis = np.asarray(list_cuis)

print("[INFO]: Statistics for Training Data")
print(f"Average number of concepts per image: {list_cuis.mean()}")
print(f"Minimum number of concepts per image: {min(list_cuis)}")
print(f"Maximum number of concepts per image: {max(list_cuis)}")

list_of_concepts_per_image = []
for i in range(max(list_cuis) + 1):
    value = int(len(np.where(list_cuis == i)[0]))
    list_of_concepts_per_image.append(value)

# Plot chart
bars = plt.bar(np.arange(0, max(list_cuis) + 1, 1), list_of_concepts_per_image)
plt.bar_label(bars)
plt.title("Training Data")
plt.xlabel('Number of Concepts')
plt.ylabel('Number of Images')
plt.xticks(np.arange(0, max(list_cuis) + 1, 1))
plt.show()

################################
#   Validation Data
################################
header = 0
df = pd.read_csv(file_name_validation, header=header, sep="\t")

cuis = df["cuis"]

list_cuis = []
for c in cuis:
    split = c.split(";")
    list_cuis.append(len(split))
list_cuis = np.asarray(list_cuis)

print("[INFO]: Statistics for Validation Data")
print(f"Average number of concepts per image: {list_cuis.mean()}")
print(f"Minimum number of concepts per image: {min(list_cuis)}")
print(f"Maximum number of concepts per image: {max(list_cuis)}")

list_of_concepts_per_image = []
for i in range(max(list_cuis) + 1):
    value = int(len(np.where(list_cuis == i)[0]))
    list_of_concepts_per_image.append(value)

# Plot chart
bars = plt.bar(np.arange(0, max(list_cuis) + 1, 1), list_of_concepts_per_image)
plt.bar_label(bars)
plt.title("Validation Data")
plt.xlabel('Number of Concepts')
plt.ylabel('Number of Images')
plt.xticks(np.arange(0, max(list_cuis) + 1, 1))
plt.show()