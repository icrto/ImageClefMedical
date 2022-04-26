# Imports
import os
import sys
import numpy as np
import pandas as pd

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# Function: Create subset according to semantic types
def get_semantic_concept_dataset(concepts_sem_csv, subset_sem_csv, semantic_type):

    # Load the .CSV concepts
    concepts_sem = pd.read_csv(concepts_sem_csv, sep="\t")
    subset_sem = pd.read_csv(subset_sem_csv, sep="\t")


    # Get the concepts that are related with the semantic type we want
    sem_type_concepts = list()
    for _, row in concepts_sem.iterrows():
        
        # Get concept_name
        concept_name = row["concept_name"]

        # Check if this concept matches the semantic type
        if concept_name.lower() == semantic_type.lower():
            sem_type_concepts.append(row["concept"])



    # Clean this list from duplicated values
    sem_type_concepts, _ = np.unique(ar=np.array(sem_type_concepts), return_counts=True)
    sem_type_concepts = list(sem_type_concepts)
    sem_type_concepts.sort()
    sem_type_concepts_dict = dict()
    
    # Create a dict for concept-mapping into classes
    for index, c in enumerate(sem_type_concepts):
        sem_type_concepts_dict[c] = index
    
    sem_type_concepts_dict[None] = index + 1


    # Get the formatted subset
    img_ids = list()
    img_labels = list()

    for index, row in subset_sem.iterrows():

        # Get image ids
        img_ids.append(row["ID"])
        
        # Get cuis
        cuis = row["cuis"]
        cuis = cuis.split(';')
        
        # Create temporary concepts list to clean subset
        tmp_concepts = list()


        # Split the cuis
        for c in cuis:
            tmp_concepts.append(c if c in sem_type_concepts else None)
        
        tmp_concepts_unique, _ = np.unique(ar=np.array(tmp_concepts), return_counts=True)
        tmp_concepts_unique = list(tmp_concepts_unique)

        if len(tmp_concepts_unique) > 1:
            label = [sem_type_concepts_dict.get(i) for i in tmp_concepts_unique]
            img_labels.append(label)
        
        else:
            label = sem_type_concepts_dict.get(tmp_concepts_unique[0])
            img_labels.append(label)



    return img_ids, img_labels



# Class: ImgClefConc Dataset
class ImgClefConcDataset(Dataset):
    def __init__(self, base_data_path, csv_path, split, random_seed=42, resized=None, low_data_regimen=None, perc_train=None, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            csv_path (string): Path for pickle with annotations.
            split (string): "train", "val", "test" splits.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        # Assure we have the right string in the split argument
        assert split in ["Train", "Validation", "Test"], "Please provide a valid split (i.e., 'Train', 'Validation' or 'Test')"

        # Aux variables to obtain the correct data splits
        # Read CSV file with label information       
        csv_df = pd.read_csv(csv_path)
        # print(f"The dataframe has: {len(csv_df)} records.")
        
        # Get the IDs of the Patients
        patient_ids = csv_df.copy()["patient_id"]
        
        # Get the unique patient ids
        unique_patient_ids = np.unique(patient_ids.values)


        # Split into train, validation and test according to the IDs of the Patients
        # First we split into train and test (60%, 20%, 20%)
        train_ids, test_ids, _, _ = train_test_split(unique_patient_ids, np.zeros_like(unique_patient_ids), test_size=0.20, random_state=random_seed)
        train_ids, val_ids, _, _ = train_test_split(train_ids, np.zeros_like(train_ids), test_size=0.25, random_state=random_seed)


        # Now, we get the data
        if split == "Train":
            # Get the right sampled dataframe
            tr_pids_mask = csv_df.copy().patient_id.isin(train_ids)
            self.dataframe = csv_df.copy()[tr_pids_mask]
            
            # Get the image names
            image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            images_labels = self.dataframe.copy()["target"].values


            # Activate low data regimen training
            if low_data_regimen:
                assert perc_train > 0.0 and perc_train <= 0.50, f"Invalid perc_train '{perc_train}'. Please be sure that perc_train > 0 and perc_train <= 50"


                # Get the data percentage
                image_names, _, images_labels, _ = train_test_split(image_names, images_labels, train_size=perc_train, stratify=images_labels, random_state=random_seed)

                print(f"Low data regimen.\n% of train data: {perc_train}")
            


            # Attribute variables object variables
            self.image_names = image_names
            self.images_labels = images_labels


            # Information print
            print(f"The {split} split has {len(self.image_names)} images")


        elif split == "Validation":
            # Get the right sampled dataframe
            val_pids_mask = csv_df.copy().patient_id.isin(val_ids)
            self.dataframe = csv_df.copy()[val_pids_mask]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.images_labels = self.dataframe.copy()["target"].values

            # Information print
            print(f"The {split} split has {len(self.image_names)} images")
        

        else:
            # Get the right sampled dataframe
            test_pids_mask = csv_df.copy().patient_id.isin(test_ids)
            self.dataframe = csv_df.copy()[test_pids_mask]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.images_labels = self.dataframe.copy()["target"].values

            # Information print
            print(f"The {split} split has {len(self.image_names)} images")


        # Init variables
        self.base_data_path = base_data_path
        # imgs_in_folder = os.listdir(self.base_data_path)
        # imgs_in_folder = [i for i in imgs_in_folder if not i.startswith(".")]
        # print(f"The folder has: {len(imgs_in_folder)} files.")

        self.transform = transform

        return


    # Method: __len__
    def __len__(self):
        return len(self.image_names)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.base_data_path, f"{img_name}.jpg"))

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label
