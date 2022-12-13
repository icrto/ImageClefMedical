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
from torchvision.models import densenet121, resnet18, DenseNet121_Weights, ResNet18_Weights

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

# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default='dataset',
                    help="Directory of the data set.")

# Model
parser.add_argument('--model', type=str, choices=["DenseNet121", "ResNet18"],
                    default="DenseNet121", help="Baseline model (DenseNet121, ResNet18).")

# Models directory
parser.add_argument('--models_dir', type=str, required=True,
                    help="Directory of the trained model(s).")

# Number of concepts
parser.add_argument("--nr_concepts", type=str, default='all',
                    help="Number of concepts to predict. Example: all, 100, etc.")

 # Eval
parser.add_argument("--images_csv", type=str, default='dataset/concept_detection_valid.csv',
                    help="csv with the images on which to run inference and generate predictions.")

# Batch size
parser.add_argument('--batchsize', type=int, default=32,
                    help="Batch-size for training and validation.")

# Image size
parser.add_argument('--imgsize', type=int, default=224,
                    help="Size of the image after transforms.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=8,
                    help="Number of workers for dataloader.")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0,
                    help="The index of the GPU.")


# Parse the arguments
args = parser.parse_args()

# Data directory
data_dir = args.data_dir

# Model
model_name = args.model.lower()

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

NR_CONCEPTS = args.nr_concepts
if NR_CONCEPTS == 'all': NR_CONCEPTS = 8374
else: NR_CONCEPTS = int(NR_CONCEPTS)

# Get data paths
if args.nr_concepts == 'all':
    sem_concepts_path = os.path.join(data_dir, "concepts_sem.csv")
else:
    sem_concepts_path = os.path.join(data_dir, f"concepts_top{args.nr_concepts}_sem.csv")

MODELS = [x for x in sorted(os.listdir(args.models_dir)) if os.path.isdir(os.path.join(args.models_dir, x))]
print(f"Processing results of {len(MODELS)} models")

# Go through semantic types and models
for semantic_type, modelckpt in zip(SEMANTIC_TYPES, MODELS):

    # Loading prints
    print(f"\nLoading the semantic type <{semantic_type}> from {modelckpt}")

    # Get nr_classes
    _, _, sem_type_concepts_dict, inv_sem_type_concepts_dict = get_semantic_concept_dataset(
        concepts_sem_csv=sem_concepts_path, subset_sem_csv=args.images_csv, semantic_type=semantic_type)

    NR_CLASSES = len(sem_type_concepts_dict)
    print(f"NR CLASSES {NR_CLASSES}")

    # Create the model object
    # DenseNet121
    if model_name == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = torch.nn.Linear(1024, NR_CLASSES)

    # ResNet18
    elif model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(512, NR_CLASSES)

    # Weights directory
    model_file = os.path.join(args.models_dir, modelckpt, "weights", "model_best.pt")

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Datasets
    data_dir = os.path.dirname(args.images_csv)
    if 'train' in args.images_csv:
        datapath = os.path.join(data_dir, 'train')
    elif 'test' in args.images_csv:
        datapath = os.path.join(data_dir, 'test')
    else:
        datapath = os.path.join(data_dir, 'valid')

    eval_set = ImgClefConcDataset(img_datapath=datapath, concepts_sem_csv=sem_concepts_path,
                                      subset_sem_csv=args.images_csv, semantic_type=semantic_type, transform=eval_transforms)

    # Dataloaders
    eval_loader = DataLoader(dataset=eval_set, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=False, num_workers=workers)

    # Create lists to append batch results
    eval_images = []
    eval_concepts = []

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for images, labels, img_ids in tqdm(eval_loader):

            # Move data data anda model to GPU (or not)
            images = images.to(DEVICE, non_blocking=True)

            # Get the logits
            logits = model(images)

            # Pass the logits through a Sigmoid to get the outputs
            outputs = torch.sigmoid(logits)
            outputs = outputs.detach().cpu()

            # Add the valid concepts
            for batch_idx in range(images.shape[0]):
                predicted_concepts = ""
                # Get the indices of the predicted concepts (# decision threshold = 0.5)
                indices = torch.where(outputs[batch_idx] >= 0.5)[0].numpy()
                for i in indices:
                    predicted_concepts += f"{inv_sem_type_concepts_dict[i]};"

                eval_images.append(img_ids[batch_idx])
                eval_concepts.append(predicted_concepts[:-1])

    # Generate metrics and .CSVs per model
    # Create a dictionary to obtain DataFrame
    eval_set = dict()
    eval_set["ID"] = eval_images
    eval_set["cuis"] = eval_concepts

    # Save this into .CSV
    evaluation_df = pd.DataFrame(data=eval_set)

    evaluation_df.to_csv(os.path.join(
        args.models_dir, f"{modelckpt}_{semantic_type}.csv"), sep="|", index=False, header=False)
