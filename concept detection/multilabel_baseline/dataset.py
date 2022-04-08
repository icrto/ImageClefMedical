import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os

class ImageDataset(Dataset):
    def __init__(self, csv_path, df_all_concepts=None, transform=None):
        self.csv_path = csv_path
        self.csv = pd.read_csv(self.csv_path, header=0, sep="\t")
        self.df_all_concepts = pd.read_csv(df_all_concepts, sep="\t") if df_all_concepts else None
        self.image_names = self.csv[:]['ID']
        self.transform = transform

        # Preprocessing labels
        self.base_dir = os.path.dirname(self.csv_path)
       
        if(self.df_all_concepts is not None):
            all_concepts = self.df_all_concepts["concept"]
            dict_concept = dict()
            for idx, c in enumerate(all_concepts):
                dict_concept[c] = idx

            matrix = np.zeros((len(self.csv["ID"]), len(all_concepts)))
            for i in range(len(self.csv["ID"])):
                dict_concepts_per_image = self.csv["cuis"][i].split(";")
                for c in dict_concepts_per_image:
                    matrix[i][dict_concept[c]] = 1

            self.labels = matrix
            print(self.labels.shape)
        else:
            self.labels = None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if 'train' in self.csv_path:
            image = Image.open(os.path.join(self.base_dir, 'train_resized', self.image_names[index] + '.jpg')).convert("RGB")
        elif 'test' in self.csv_path:
            image = Image.open(os.path.join(self.base_dir, 'test_resized', self.image_names[index] + '.jpg')).convert("RGB")
        else:
            image = Image.open(os.path.join(self.base_dir, 'valid_resized', self.image_names[index] + '.jpg')).convert("RGB")
        
        # apply image transforms
        if(self.transform):
            image = self.transform(image)

        if(self.labels is not None):
            targets = self.labels[index]

            return {
                'image': image,
                'label': torch.tensor(targets, dtype=torch.float32)
            }
        else:
            return {
                'image': image
            }