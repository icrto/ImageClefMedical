# Imports
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import datetime

# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import densenet121, resnet18
from torch.utils.tensorboard import SummaryWriter

# Project Imports
from data_utilities import get_semantic_concept_dataset, ImgClefConcDataset
from model_utilities import freeze_feature_extractor, unfreeze_feature_extractor

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, required=True,
                    help="Directory of the data set.")

# Model
parser.add_argument('--model', type=str, choices=["DenseNet121", "ResNet18"],
                    default="ResNet18", help="Backbone model to train: DenseNet121 or ResNet18.")

# Semantic type
parser.add_argument('--semantic_type', type=str, required=True, choices=["Body Part, Organ, or Organ Component", "Spatial Concept", "Finding", "Pathologic Function", "Qualitative Concept", "Diagnostic Procedure", "Body Location or Region", "Functional Concept",
                    "Miscellaneous Concepts"], help='Semantic type:"Body Part, Organ, or Organ Component", "Spatial Concept", "Finding", "Pathologic Function", "Qualitative Concept", "Diagnostic Procedure", "Body Location or Region", "Functional Concept", "Miscellaneous Concepts".')

# Batch size
parser.add_argument('--batchsize', type=int, default=4,
                    help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224,
                    help="Size of the image after transforms")

# Class Weights
parser.add_argument("--classweights", action="store_true",
                    help="Weight loss with class imbalance")

# Freeze backbone
parser.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze backbone at training start")

# Number of epochs
parser.add_argument('--epochs', type=int, default=20,
                    help="Number of training epochs")
parser.add_argument('--epochs_freeze', type=int, default=5,
                    help="Number of training epochs where backbone is frozen")

# Learning rate
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

# Output directory
parser.add_argument("--outdir", type=str, default="results",
                    help="Output directory")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0,
                    help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0,
                    help="The index of the GPU")

# Save frequency
parser.add_argument("--save_freq", type=int, default=10,
                    help="Frequency (in number of epochs) to save the model")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint from which to resume training")


# Parse the arguments
args = parser.parse_args()


# Resume training
if args.resume:
    assert args.ckpt is not None, "Please specify the model checkpoint when resume is True"

resume = args.resume

# Training checkpoint
ckpt = args.ckpt


# Data directory
data_dir = args.data_dir

# Semantic type
semantic_type = args.semantic_type

# Results Directory
outdir = args.outdir

# Number of workers (threads)
workers = args.num_workers

# Number of training epochs
EPOCHS = args.epochs

EPOCHS_FREEZE = args.epochs_freeze

# Learning rate
LEARNING_RATE = args.lr

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Save frquency
save_freq = args.save_freq

# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = os.path.join(outdir, "semantic_baseline", timestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)


# Save training parameters
with open(os.path.join(outdir, "train_params.txt"), "w") as f:
    f.write(str(args))


# Results and Weights
weights_dir = os.path.join(outdir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(outdir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


# Tensorboard
tbwritter = SummaryWriter(log_dir=os.path.join(
    outdir, "tensorboard"), flush_secs=5)


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
sem_concepts_path = os.path.join(
    data_dir, "csv", "concepts", "top100", "new_top100_concepts_sem.csv")

train_datapath = os.path.join(data_dir, "dataset_resized", "train_resized")
train_csvpath = os.path.join(
    data_dir, "csv", "concepts", "top100", "new_train_subset_top100_sem.csv")

valid_datapath = os.path.join(data_dir, "dataset_resized", "valid_resized")
valid_csvpath = os.path.join(
    data_dir, "csv", "concepts", "top100", "new_val_subset_top100_sem.csv")

# Get nr_classes
_, _, sem_type_concepts_dict, _ = get_semantic_concept_dataset(
    concepts_sem_csv=sem_concepts_path, subset_sem_csv=train_csvpath, semantic_type=semantic_type)

NR_CLASSES = len(sem_type_concepts_dict)
print(f"SEMANTIC TYPE: {semantic_type}")
print(f"NR CLASSES {NR_CLASSES}")


# Choose model(s)
model_name = args.model.lower()
# DenseNet121
if model_name == "densenet121".lower():

    model = densenet121(progress=True, pretrained=True)
    model.classifier = torch.nn.Linear(1024, NR_CLASSES)

# ResNet18
elif model_name == "resnet18".lower():
    model = resnet18(progress=True, pretrained=True)
    model.fc = torch.nn.Linear(512, NR_CLASSES)


# Put model into DEVICE (CPU or GPU)
model = model.to(DEVICE)


# Load data
# Train
# Transforms
train_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=(-10, 10), translate=(
        0.05, 0.1), scale=(0.95, 1.05), shear=0, fill=(0, 0, 0)),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation
# Transforms
valid_transforms = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Data sets and class weights for loss function
classweights = args.classweights
train_set = ImgClefConcDataset(img_datapath=train_datapath, concepts_sem_csv=sem_concepts_path,
                               subset_sem_csv=train_csvpath, semantic_type=semantic_type, transform=train_transforms, classweights=classweights)
valid_set = ImgClefConcDataset(img_datapath=valid_datapath, concepts_sem_csv=sem_concepts_path,
                               subset_sem_csv=valid_csvpath, semantic_type=semantic_type, transform=valid_transforms)
cw = train_set.pos_weights
if cw is not None:
    cw = torch.from_numpy(cw).to(DEVICE)
print(f"Using class weights: cw={cw}")

# Hyper-parameters
LOSS = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=cw)
VAL_LOSS = torch.nn.BCEWithLogitsLoss(reduction="sum")


# Freeze backbone if needed to warm-up the training
freeze_backbone = args.freeze_backbone
if freeze_backbone:
    freeze_feature_extractor(model=model, name=model_name)
    OPTIMISER = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    SCHEDULER = None

else:
    OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    SCHEDULER = torch.optim.lr_scheduler.LambdaLR(
        OPTIMISER, lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)

# Resume training from given checkpoint
if resume:
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    OPTIMISER.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from {ckpt} at epoch {init_epoch}")
else:
    init_epoch = 0


# Dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                          shuffle=True, pin_memory=False, num_workers=workers)
val_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE,
                        shuffle=True, pin_memory=False, num_workers=workers)


# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Go through the number of Epochs
for epoch in range(init_epoch, EPOCHS):
    # Epoch
    print(f"Epoch: {epoch+1}")

    # Training Loop
    print("Training Phase")

    # Running train loss
    run_train_loss = torch.tensor(0, dtype=torch.float64, device=DEVICE)

    # Put model in training mode
    model.train()

    # If we started with frozen backbone and reach EPOCHS_FREEZE
    # We unfreeze the models
    if freeze_backbone and (epoch + 1) >= EPOCHS_FREEZE:
        print("Feature extractor will be trainable from now on...")
        unfreeze_feature_extractor(model=model, name=model_name)
        OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        SCHEDULER = torch.optim.lr_scheduler.LambdaLR(
            OPTIMISER, lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)
        freeze_backbone = False
        print("Unfreezing complete.")

    # Iterate through dataloader
    for images, labels, _ in tqdm(train_loader):

        # Move data and model to GPU (or not)
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(
            DEVICE, non_blocking=True)

        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad(set_to_none=True)

        # Get logits
        logits = model(images)

        # Compute the batch loss
        loss = LOSS(logits, labels)

        # Update batch losses
        run_train_loss += loss.item()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        OPTIMISER.step()

    # update scheduler at each epoch
    if SCHEDULER:
        SCHEDULER.step()

    # Compute Average Train Loss
    avg_train_loss = run_train_loss / len(train_loader.dataset)

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}")

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", avg_train_loss, global_step=epoch)
    if SCHEDULER:
        tbwritter.add_scalar("lr", SCHEDULER.get_last_lr()[
            0], global_step=epoch)

    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(
            f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss

    # Validation Loop
    print("Validation Phase")

    # Running train loss
    run_val_loss = 0.0

    # Put model in evaluation mode
    model.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for images, labels, _ in tqdm(val_loader):

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(
                DEVICE, non_blocking=True)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)

            # Compute the batch loss
            loss = VAL_LOSS(logits, labels)

            # Update batch losses
            run_val_loss += loss.item()

        # Compute Average Validation Loss
        avg_val_loss = run_val_loss / len(val_loader.dataset)

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss}")

        # Plot to Tensorboard
        tbwritter.add_scalar("loss/val", avg_val_loss, global_step=epoch)

        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(
                f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            print("Saving best model on validation...")

            # Save checkpoint
            model_path = os.path.join(
                weights_dir, f"model_best.pt")

            if SCHEDULER:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': OPTIMISER.state_dict(),
                    'sched_state_dict': SCHEDULER.state_dict(),
                    'loss': avg_train_loss,
                }
            else:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': OPTIMISER.state_dict(),
                    'loss': avg_train_loss,
                }
            torch.save(save_dict, model_path)

            print(f"Successfully saved at: {model_path}")

        # Checkpoint loop/condition
        if epoch % save_freq == 0 and epoch > 0:

            # Save checkpoint
            model_path = os.path.join(
                weights_dir, f"model_{epoch:04}.pt")

            if SCHEDULER:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': OPTIMISER.state_dict(),
                    'sched_state_dict': SCHEDULER.state_dict(),
                    'loss': avg_train_loss,
                }
            else:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': OPTIMISER.state_dict(),
                    'loss': avg_train_loss,
                }
            torch.save(save_dict, model_path)


# Finish statement
print("Finished.")
