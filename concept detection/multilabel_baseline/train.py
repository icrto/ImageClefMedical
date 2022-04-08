import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from baseline import model, train, validate
from dataset import ImageDataset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sklearn.metrics
import numpy as np
import pandas as pd

# Arguments
BASE_DIR = '/BARRACUDA8T/ImageCLEF2022/dataset_resized'
NR_CONCEPTS = 100
FIGURE_NAME = f"Fig_Loss_DenseNet121_{NR_CONCEPTS}"
MODEL_TO_SAVE_NAME = f"model_densenet121_test_{NR_CONCEPTS}"
TRAIN_FE = True # change to False to freeze entire feature extraction backbone
WORKERS = 12
IMG_SIZE = (224, 224)

matplotlib.style.use('ggplot')

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
model = model(pretrained=True, requires_grad=TRAIN_FE, nr_concepts=NR_CONCEPTS).to(device)

# learning parameters
lr = 0.0001
epochs = 15
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# read the training csv file
if(NR_CONCEPTS == 100):
    train_csv = os.path.join(BASE_DIR, 'new_train_subset_top100.csv')
    val_csv = os.path.join(BASE_DIR, 'new_val_subset_top100.csv')
    concept_dict = os.path.join(BASE_DIR, "new_top100_concepts.csv")
else:
    train_csv = os.path.join(BASE_DIR, 'concept_detection_train.csv')
    val_csv = os.path.join(BASE_DIR, 'concept_detection_valid.csv')
    concept_dict = os.path.join(BASE_DIR, "new_top100_concepts.csv")

# train dataset
train_transform = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE),
                #transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
train_data = ImageDataset(
    train_csv, train=True, df_all_concepts=concept_dict, transform=train_transform
)

# validation dataset
val_transform = transforms.Compose([
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
valid_data = ImageDataset(
    val_csv, train=False, df_all_concepts=concept_dict, transform=val_transform
)

# train data loader
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=WORKERS
)

# validation data loader
valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=WORKERS

)

# start the training and validation
train_loss = []
valid_loss = []
best_val_loss = np.inf
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, device
    )
    valid_epoch_loss = validate(
        model, valid_loader, criterion, device
    )
    if valid_epoch_loss < best_val_loss:
        best_val_loss = valid_epoch_loss
        # save model with best val loss
        # torch.save({
        #     'epoch': epochs,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': valid_epoch_loss,
        # }, f'{MODEL_TO_SAVE_NAME}_best.pth')

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')

# save the trained model to disk
# torch.save({
#             'epoch': epochs,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': valid_epoch_loss,
# }, f'{MODEL_TO_SAVE_NAME}_last.pth')

# # plot and save the train and validation line graphs
# plt.figure(figsize=(10, 7))
# plt.plot(train_loss, color='orange', label='train loss')
# plt.plot(valid_loss, color='red', label='validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig(f'{FIGURE_NAME}.png')
# plt.show()