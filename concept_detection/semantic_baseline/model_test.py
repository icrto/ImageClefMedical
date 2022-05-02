# Imports
import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import densenet121, resnet18

# Sklearn Imports
import sklearn.metrics

# Project Imports
from data_utilities import get_semantic_concept_dataset, ImgClefConcDataset


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Semantic types
SEMANTIC_TYPES = [
    "Body Part, Organ, or Organ Component",
    "Spatial Concept",
    "Finding",
    "Pathologic Function",
    "Qualitative Concept",
    "Diagnostic Procedure",
    "Body Location or Region",
    "Functional Concept",
    "Miscellaneous Concepts"
]


# Models
MODELS = [
    "2022-05-01_01-29-42",
    "2022-05-01_01-55-28",
    "2022-05-01_02-21-17",
    "2022-05-01_02-47-06",
    "2022-05-01_03-12-56",
    "2022-05-01_03-38-44",
    "2022-05-01_04-04-33",
    "2022-05-01_04-30-22",
    "2022-05-01_04-56-11"
]



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")

# Subset
parser.add_argument('--subset', type=str, choices=["train", "validation", "test"], default="validation", help="Subset of data.")

# Model
parser.add_argument('--model', type=str, choices=["densenet121", "resnet18"], default="resnet18", help="Baseline model (DenseNet121, ResNet18).")

# Batch size
parser.add_argument('--batchsize', type=int, default=1, help="Batch-size for training and validation.")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU.")


# Parse the arguments
args = parser.parse_args()

# Data directory
data_dir = args.data_dir

# Subset
subset = args.subset

# Model
model_name = args.model

# Number of workers (threads)
workers = args.num_workers

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE

# Get data paths
sem_concepts_path = os.path.join(data_dir, "csv", "concepts", "top100", "new_top100_concepts_sem.csv")

# Train
train_datapath = os.path.join(data_dir, "dataset_resized", "train_resized")
train_csvpath = os.path.join(data_dir, "csv", "concepts", "top100", "new_train_subset_top100_sem.csv")

# Validation
valid_datapath = os.path.join(data_dir, "dataset_resized", "valid_resized")
valid_csvpath = os.path.join(data_dir, "csv", "concepts", "top100", "new_val_subset_top100_sem.csv")

# Test
test_datapath = os.path.join(data_dir, "dataset_resized", "test_resized")
test_csvpath = os.path.join(data_dir, "csv", "test_images.csv")



# Go through semantic types and models
for semantic_type, modelckpt in zip(SEMANTIC_TYPES, MODELS):

    # Loading prints
    print(f"Loading the semantic type <{semantic_type}> from {modelckpt}")

    # Get nr_classes
    _, _, sem_type_concepts_dict, inv_sem_type_concepts_dict = get_semantic_concept_dataset(concepts_sem_csv=sem_concepts_path, subset_sem_csv=train_csvpath, semantic_type=semantic_type)

    NR_CLASSES = len(sem_type_concepts_dict)
    print(f"NR CLASSES {NR_CLASSES}")


    # Create the model object
    if model_name.lower() == "densenet121":
        model = densenet121(progress=True, pretrained=True)
        model.classifier = torch.nn.Linear(1024, NR_CLASSES)

    elif model_name.lower() == "resnet18":
        model = resnet18(progress=True, pretrained=True)
        model.fc = torch.nn.Linear(512, NR_CLASSES)


    # Weights directory
    weights_dir = os.path.join("results", "semantic_baseline", "model_checkpoints", modelckpt, "weights")
    model_file = os.path.join(weights_dir, "model_best.pt")

    # Load model weights
    checkpoint = torch.load(model_file, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"Loaded model from {model_file}")

    # Put model into DEVICE (CPU or GPU)
    model = model.to(DEVICE)


    # Put model into evaluation mode
    model.eval()


    # Load data
    # Test Transforms
    eval_transforms = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Datasets
    if subset == "test":
        eval_set = ImgClefConcDataset(img_datapath=test_datapath, concepts_sem_csv=sem_concepts_path, subset_sem_csv=test_csvpath, semantic_type=semantic_type, transform=eval_transforms, subset=subset)

    else:
        eval_set = ImgClefConcDataset(img_datapath=valid_datapath, concepts_sem_csv=sem_concepts_path, subset_sem_csv=valid_csvpath, semantic_type=semantic_type, transform=eval_transforms)


    # Dataloaders
    eval_loader = DataLoader(dataset=eval_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=workers)



    # Create lists to append batch results
    y_true = []
    y_pred = []
    eval_images = []
    eval_concepts = []

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for images, labels, img_ids in tqdm(eval_loader):

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # Get the logits
            logits = model(images)

            # Pass the logits through a Sigmoid to get the outputs
            outputs = torch.sigmoid(logits)
            outputs = outputs.detach().cpu()

            # Get the indices of the predicted concepts (# decision threshold = 0.5)
            indices = np.where(outputs.numpy()[0] >= 0.5)

            # Add the valid concepts
            predicted_concepts = ""
            for i in indices[0]:
                predicted_concepts += f"{inv_sem_type_concepts_dict[i]};"
            
            eval_images.append(img_ids[0])
            eval_concepts.append(predicted_concepts[:-1])

            if len(labels) > 0:
                zero_array = np.zeros_like(labels[0].cpu().detach().numpy())
                for idx in indices:
                    zero_array[idx] = 1

                y_pred.append(zero_array)
                y_true.append(labels[0].numpy())
    

    # Generate metrics and .CSVs per model
    # Create a dictionary to obtain DataFrame
    eval_set = dict()
    eval_set["ID"] = eval_images
    eval_set["cuis"] = eval_concepts

    # Save this into .CSV
    evaluation_df = pd.DataFrame(data=eval_set)

    if len(y_true) > 0:
        print(f"/////////// Evaluation Report ////////////")
        print(f"Exact Match Ratio: {sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):.4f}")
        print(f"Hamming loss: {sklearn.metrics.hamming_loss(y_true, y_pred):.4f}")
        print(f"Recall: {sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        print(f"Precision: {sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        print(f"F1 Measure: {sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        
        save_dir = os.path.join("results", "semantic_baseline", "validation")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        evaluation_df.to_csv(os.path.join("semantic_baseline", "validation", f"{semantic_type}.csv"), sep="\t", index=False)
    
    else:
        save_dir = os.path.join("results", "semantic_baseline", "test")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        evaluation_df.to_csv(os.path.join(save_dir, f"{semantic_type}.csv"), sep="|", index=False, header=False)



    # Finish statement
    print("Finished.")
