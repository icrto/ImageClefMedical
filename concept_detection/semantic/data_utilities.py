# Imports
import os
import numpy as np
import pandas as pd
import PIL

# PyTorch Imports
import torch
from torch.utils.data import Dataset

# Function: Create subset according to semantic types
def get_semantic_concept_dataset(concepts_sem_csv, subset_sem_csv, semantic_type, **kwargs):

    assert semantic_type in ("Body Part, Organ, or Organ Component", "Spatial Concept", "Finding", "Pathologic Function", "Qualitative Concept", "Diagnostic Procedure",
                             "Body Location or Region", "Functional Concept", "Miscellaneous Concepts"), f"{semantic_type} not valid. Provide a valida semantic type"

    # Load the .CSV concepts
    concepts_sem = pd.read_csv(concepts_sem_csv, sep="\t")

    if "subset" in kwargs.keys():
        if kwargs['subset'].lower() == "test":
            subset_sem = pd.read_csv(subset_sem_csv)
    
    else:
        subset_sem = pd.read_csv(subset_sem_csv, sep="\t")

    # Get the concepts that are related with the semantic type we want
    sem_type_concepts = list()
    for _, row in concepts_sem.iterrows():

        # Get concept_name
        concept_name = row["semantic_types"]

        # Check if this concept matches the semantic type
        if concept_name in ("Body Part, Organ, or Organ Component", "Spatial Concept", "Finding", "Pathologic Function", "Qualitative Concept", "Diagnostic Procedure", "Body Location or Region", "Functional Concept"):
            if concept_name == semantic_type:
                sem_type_concepts.append(row["concept"])

        elif semantic_type == "Miscellaneous Concepts":
            sem_type_concepts.append(row["concept"])

    # Clean this list from duplicated values
    sem_type_concepts, _ = np.unique(ar=np.array(sem_type_concepts), return_counts=True)
    sem_type_concepts = list(sem_type_concepts)
    sem_type_concepts.sort()
    sem_type_concepts_dict = dict()
    inv_sem_type_concepts_dict = dict()

    # Create a dict for concept-mapping into classes
    for index, c in enumerate(sem_type_concepts):
        sem_type_concepts_dict[c] = index
        inv_sem_type_concepts_dict[index] = c

    # Get the formatted subset
    img_ids = list()
    img_labels = list()

    for index, row in subset_sem.iterrows():

        # Get image ids
        img_ids.append(row["ID"])

        # Get cuis
        if "subset" in kwargs.keys():
            if kwargs['subset'].lower() == "test":
                img_labels = list()
        
        else:
            cuis = row["cuis"]
            cuis = cuis.split(';')

            # Create temporary concepts list to clean subset
            tmp_concepts = list()

            # Split the cuis
            for c in cuis:
                if c in sem_type_concepts:
                    tmp_concepts.append(c)

            tmp_concepts_unique, _ = np.unique(ar=np.array(tmp_concepts), return_counts=True)
            tmp_concepts_unique = list(tmp_concepts_unique)

            if len(tmp_concepts_unique) > 0:
                label = [sem_type_concepts_dict.get(i) for i in tmp_concepts_unique]
                img_labels.append(label)

            else:
                label = []
                img_labels.append(label)


    # Test Set
    if "subset" in kwargs.keys():
        if kwargs['subset'].lower() == "test":
            img_labels = list()
    
    else:
        pass


    return img_ids, img_labels, sem_type_concepts_dict, inv_sem_type_concepts_dict



# Class: ImgClefConc Dataset
class ImgClefConcDataset(Dataset):
    def __init__(self, img_datapath, concepts_sem_csv, subset_sem_csv, semantic_type, transform=None, subset=None, classweights=None):

        # Get the desired dataset
        if subset:
            self.img_ids, img_labels, self.sem_type_concepts_dict, self.inv_sem_type_concepts_dict = get_semantic_concept_dataset(concepts_sem_csv=concepts_sem_csv, subset_sem_csv=subset_sem_csv, semantic_type=semantic_type, subset=subset)

        else:
            self.img_ids, img_labels, self.sem_type_concepts_dict, self.inv_sem_type_concepts_dict = get_semantic_concept_dataset(concepts_sem_csv=concepts_sem_csv, subset_sem_csv=subset_sem_csv, semantic_type=semantic_type)

        self.img_datapath = img_datapath
        self.transform = transform

        # Since we are dealing with a multilabel case
        if len(img_labels) > 0:
            matrix_labels = np.zeros((len(self.img_ids), len(self.sem_type_concepts_dict)))
            for r in range(len(self.img_ids)):
                label = img_labels[r]

                for c in label:
                    matrix_labels[r, c] = 1

            self.img_labels = matrix_labels.copy()


            # Compute class weights for loss function
            if classweights:
                
                # pos_weights = neg / pos
                pos_count = np.count_nonzero(matrix_labels.copy(), axis=0)
                neg_count = len(matrix_labels) - pos_count

                np.testing.assert_array_equal(np.ones_like(pos_count) * len(matrix_labels), np.sum((neg_count, pos_count), axis=0))
                pos_weights = neg_count / pos_count

                self.pos_weights = pos_weights
            
            else:
                self.pos_weights = None
        

        else:
            self.img_labels = img_labels


        return


    # Method: __len__

    def __len__(self):
        return len(self.img_ids)

    # Method: __getitem__

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get images
        img_name = self.img_ids[idx]
        image = PIL.Image.open(os.path.join(self.img_datapath, f"{img_name}.jpg")).convert("RGB")

        # Get labels
        if len(self.img_labels) > 0:
            label = self.img_labels[idx]
        
        else:
            label = list()
            label = torch.Tensor(label)

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label, img_name



# Run this script to test code
if __name__ == "__main__":

    # Get data paths
    data_path = "dataset"
    sem_concepts = os.path.join(data_path, "concepts_top100_sem.csv")

    train_csv = os.path.join(data_path, "concept_detection_train_top100.csv")

    print("Processing...")
    img_ids, img_labels, sem_type_concepts_dict, inv_sem_type_concepts_dict = get_semantic_concept_dataset(concepts_sem_csv=sem_concepts, subset_sem_csv=train_csv, semantic_type="Miscellaneous Concepts")

    print(img_ids, img_labels)
