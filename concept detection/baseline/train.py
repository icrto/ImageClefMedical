import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from baseline import model, train, validate
from baseline import ImageDataset

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sklearn.metrics
from tqdm import tqdm

# Arguments
FIGURE_NAME = "Fig_Loss_DenseNet121"
MODEL_TO_SAVE_NAME = "model_densenet121_test"

matplotlib.style.use('ggplot')

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
model = model(pretrained=True, requires_grad=False).to(device)

# learning parameters
lr = 0.0001
epochs = 15
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# read the training csv file
train_csv = pd.read_csv('../../data/dataset_resized/concept_detection_train.csv', header=0, sep="\t")
#train_csv = pd.read_csv('../../data/dataset_resized/new_train_subset_top100.csv', header=0, sep="\t")

# train dataset
train_data = ImageDataset(
    train_csv, train=True, test=False
)

# validation dataset
valid_data = ImageDataset(
    train_csv, train=False, test=False
)

# train data loader
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

# validation data loader
valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False
)

# start the training and validation
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, train_data, device
    )
    valid_epoch_loss = validate(
        model, valid_loader, criterion, valid_data, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')

# save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f'{MODEL_TO_SAVE_NAME}.pth')

# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{FIGURE_NAME}.png')
plt.show()