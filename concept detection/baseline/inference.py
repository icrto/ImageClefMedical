import torch
from torch.utils.data import DataLoader
from baseline import model
from baseline import ImageDataset

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

import sklearn.metrics

# Directories and Files
DATA_DIR = "."
TOP_K_CONCEPTS = 100

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# intialize the model
model = model(pretrained=False, requires_grad=False).to(device)

# load the model checkpoint
checkpoint = torch.load('../../models/model_densenet121.pth')

# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#test_csv = pd.read_csv('../../data/dataset_resized/concept_detection_valid.csv', header=0, sep="\t")
test_csv = pd.read_csv('../../data/dataset_resized/new_val_subset_top100.csv', header=0, sep="\t")

image_ids = test_csv["ID"]

# prepare the test dataset and dataloader
test_data = ImageDataset(
    test_csv, train=False, test=True
)

test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False
)

# Preprocessing labels
df_all_concepts = pd.read_csv("../../data/dataset_resized/concepts.csv",
                              sep="\t")
#df_all_concepts = pd.read_csv("../../data/dataset_resized/new_top100_concepts.csv", sep="\t")
all_concepts = df_all_concepts["concept"].tolist()

y_true = []
y_pred = []
eval_images = []
eval_concepts = []
for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']

    y_true.append(target[0].numpy())

    # get the predictions by passing the image through the model
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()

    indices = np.where(outputs.numpy()[0] >= 0.5)  # decision threshold = 0.5

    # Add the valid concepts
    predicted_concepts = ""
    for i in indices[0]:
        predicted_concepts += f"{all_concepts[i]};"

    eval_images.append(image_ids[counter])
    eval_concepts.append(predicted_concepts[:-1])

    zero_array = np.zeros_like(target[0])
    for idx in indices:
        zero_array[idx] = 1

    y_pred.append(zero_array)

# Generate Evaluation CSV
# Create a dictionary to obtain DataFrame
eval_set = dict()
eval_set["ID"] = eval_images
eval_set["cuis"] = eval_concepts

# Save this into .CSV
evaluation_df = pd.DataFrame(data=eval_set)
evaluation_df.to_csv(os.path.join(DATA_DIR, f"eval_set_top_{TOP_K_CONCEPTS}_test_time.csv"), sep="\t", index=False)

print(f"/////////// Evaluation Report ////////////")
print(f"Exact Match Ratio: {sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):.4f}")
print(f"Hamming loss: {sklearn.metrics.hamming_loss(y_true, y_pred):.4f}")
print(f"Recall: {sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
print(f"Precision: {sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
print(f"F1 Measure: {sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")