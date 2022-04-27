# Imports
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from torchinfo import summary

# Sklearn Imports
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


# Project Imports
from model_utilities import DenseNet121
from data_utilities import get_semantic_concept_dataset, ImgClefConcDataset



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")

# Semantic type
parser.add_argument('--semantic_type', type=str, required=True, choices=["Body Part, Organ, or Organ Component", "Spatial Concept", "Finding", "Pathologic Function", "Qualitative Concept", "Diagnostic Procedure", "Body Location or Region", "Functional Concept", "Miscellaneous Concepts"], help='Semantic type:"Body Part, Organ, or Organ Component", "Spatial Concept", "Finding", "Pathologic Function", "Qualitative Concept", "Diagnostic Procedure", "Body Location or Region", "Functional Concept", "Miscellaneous Concepts".')

# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121"], help='Model Name: DenseNet121.')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Class Weights
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Number of epochs
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

# Learning rate
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

# Output directory
parser.add_argument("--outdir", type=str, default="results", help="Output directory")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Save frequency
parser.add_argument("--save_freq", type=int, default=10, help="Frequency (in number of epochs) to save the model")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint from which to resume training")



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

# Learning rate
LEARNING_RATE = args.lr

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Save frquency
save_freq = args.save_freq

# Resize (data transforms)
resize_opt = args.resize
model = args.model
model_name = model.lower()



# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = os.path.join(outdir, "semantic_baseline", model_name, timestamp)
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
tbwritter = SummaryWriter(log_dir=os.path.join(outdir, "tensorboard"), flush_secs=30)


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

train_datapath = os.path.join(data_dir, "dataset_resized", "train_resized")
train_csvpath = os.path.join(data_dir, "csv", "concepts", "top100", "new_train_subset_top100_sem.csv")

valid_datapath = os.path.join(data_dir, "dataset_resized", "valid_resized")
valid_csvpath = os.path.join(data_dir, "csv", "concepts", "top100", "new_val_subset_top100_sem.csv")

# Get nr_classes
_, _, sem_type_concepts_dict = get_semantic_concept_dataset(concepts_sem_csv=sem_concepts_path, subset_sem_csv=train_csvpath, semantic_type=semantic_type)

NR_CLASSES = len(sem_type_concepts_dict)



# DenseNet-121
if model == "DenseNet121":
    model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=NR_CLASSES)



# Put model into DEVICE (CPU or GPU)
model = model.to(DEVICE)


# Get model summary
try:
    model_summary = summary(model, (1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)

except:
    model_summary = str(model)


# Write into file
with open(os.path.join(outdir, "model_summary.txt"), 'w') as f:
    f.write(str(model_summary))



# Load data
# Train
# Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Validation
# Transforms
valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Datasets
train_set = ImgClefConcDataset(img_datapath=train_datapath, concepts_sem_csv=sem_concepts_path, subset_sem_csv=train_csvpath, semantic_type=semantic_type, transform=train_transforms)
valid_set = ImgClefConcDataset(img_datapath=valid_datapath, concepts_sem_csv=sem_concepts_path, subset_sem_csv=valid_csvpath, semantic_type=semantic_type, transform=valid_transforms)



# Class weights for loss
if args.classweights:
    classes = np.array(range(NR_CLASSES))
    cw = compute_class_weight('balanced', classes=classes, y=np.array(train_set.img_labels))
    cw = torch.from_numpy(cw).float().to(DEVICE)
    print(f"Using class weights {cw}")
else:
    cw = None


# Hyper-parameters
LOSS = torch.nn.CrossEntropyLoss(reduction="sum", weight=cw)
VAL_LOSS = torch.nn.CrossEntropyLoss(reduction="sum")
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



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
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=workers)
val_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=workers)



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Initialise losses arrays
train_losses = np.zeros((EPOCHS, ))
val_losses = np.zeros_like(train_losses)

# Initialise metrics arrays
train_metrics = np.zeros((EPOCHS, 5))
val_metrics = np.zeros_like(train_metrics)

# Go through the number of Epochs
for epoch in range(init_epoch, EPOCHS):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Loop
    print("Training Phase")
    
    # Initialise lists to compute scores
    y_train_true = np.empty((0), int)
    y_train_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)
    y_train_scores = torch.empty(0, dtype=torch.float, device=DEVICE) # save scores after softmax for roc auc


    # Running train loss
    run_train_loss = torch.tensor(0, dtype=torch.float64, device=DEVICE)


    # Put model in training mode
    model.train()

    # Iterate through dataloader
    for images, labels in tqdm(train_loader):
        # Concatenate lists
        y_train_true = np.append(y_train_true, labels.numpy(), axis=0)

        # Move data and model to GPU (or not)
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad(set_to_none=True)


        # Get logits
        logits = model(images)

        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)

        # Update batch losses
        run_train_loss += loss

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()

        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        # s_logits = torch.nn.Softmax(dim=1)(logits)
        # y_train_scores = torch.cat((y_train_scores, s_logits))
        # s_logits = torch.argmax(s_logits, dim=1)
        # y_train_pred = torch.cat((y_train_pred, s_logits))


    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)
    

    # Compute Train Metrics
    # y_train_pred = y_train_pred.cpu().detach().numpy()
    # y_train_scores = y_train_scores.cpu().detach().numpy()
    # train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
    # train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    # train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    # train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    # train_auc = roc_auc_score(y_true=y_train_true, y_score=y_train_scores[:, 1], average='micro')

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}")
    # print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}")
    # print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}\tTrain Recall: {train_recall}\tTrain Precision: {train_precision}\tTrain F1-Score: {train_f1}")


    # Append values to the arrays
    # Train Loss
    train_losses[epoch] = avg_train_loss
    # Save it to directory
    fname = os.path.join(history_dir, f"{model_name}_tr_losses.npy")
    np.save(file=fname, arr=train_losses, allow_pickle=True)


    # Train Metrics
    # Acc
    # train_metrics[epoch, 0] = train_acc
    # Recall
    # train_metrics[epoch, 1] = train_recall
    # Precision
    # train_metrics[epoch, 2] = train_precision
    # F1-Score
    # train_metrics[epoch, 3] = train_f1
    # ROC AUC
    # train_metrics[epoch, 4] = train_auc

    # Save it to directory
    fname = os.path.join(history_dir, f"{model_name}_tr_metrics.npy")
    np.save(file=fname, arr=train_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", avg_train_loss, global_step=epoch)
    # tbwritter.add_scalar("acc/train", train_acc, global_step=epoch)
    # tbwritter.add_scalar("rec/train", train_recall, global_step=epoch)
    # tbwritter.add_scalar("prec/train", train_precision, global_step=epoch)
    # tbwritter.add_scalar("f1/train", train_f1, global_step=epoch)
    # tbwritter.add_scalar("auc/train", train_auc, global_step=epoch)

    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss


    # Validation Loop
    print("Validation Phase")


    # Initialise lists to compute scores
    y_val_true = np.empty((0), int)
    y_val_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)
    y_val_scores = torch.empty(0, dtype=torch.float, device=DEVICE) # save scores after softmax for roc auc

    # Running train loss
    run_val_loss = 0.0

    # Put model in evaluation mode
    model.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for images, labels in tqdm(val_loader):
            y_val_true = np.append(y_val_true, labels.numpy(), axis=0)

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = VAL_LOSS(logits, labels)
            
            # Update batch losses
            run_val_loss += loss


            # Using Softmax Activation
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            # s_logits = torch.nn.Softmax(dim=1)(logits)                        
            # y_val_scores = torch.cat((y_val_scores, s_logits))
            # s_logits = torch.argmax(s_logits, dim=1)
            # y_val_pred = torch.cat((y_val_pred, s_logits))

        

        # Compute Average Validation Loss
        avg_val_loss = run_val_loss/len(val_loader.dataset)

        # Compute Validation Accuracy
        # y_val_pred = y_val_pred.cpu().detach().numpy()
        # y_val_scores = y_val_scores.cpu().detach().numpy()
        # val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)
        # val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        # val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        # val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        # val_auc = roc_auc_score(y_true=y_val_true, y_score=y_val_scores[:, 1], average='micro')

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss}")
        # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}")
        # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

        # Append values to the arrays
        # Validation Loss
        val_losses[epoch] = avg_val_loss
        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_losses.npy")
        np.save(file=fname, arr=val_losses, allow_pickle=True)


        # Train Metrics
        # Acc
        # val_metrics[epoch, 0] = val_acc
        # Recall
        # val_metrics[epoch, 1] = val_recall
        # Precision
        # val_metrics[epoch, 2] = val_precision
        # F1-Score
        # val_metrics[epoch, 3] = val_f1
        # ROC AUC
        # val_metrics[epoch, 4] = val_auc

        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_metrics.npy")
        np.save(file=fname, arr=val_metrics, allow_pickle=True)

        # Plot to Tensorboard
        tbwritter.add_scalar("loss/val", avg_val_loss, global_step=epoch)
        # tbwritter.add_scalar("acc/val", val_acc, global_step=epoch)
        # tbwritter.add_scalar("rec/val", val_recall, global_step=epoch)
        # tbwritter.add_scalar("prec/val", val_precision, global_step=epoch)
        # tbwritter.add_scalar("f1/val", val_f1, global_step=epoch)
        # tbwritter.add_scalar("auc/val", val_auc, global_step=epoch)

        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            print("Saving best model on validation...")

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"{model_name}_{semantic_type}_best.pt")
            
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
            model_path = os.path.join(weights_dir, f"{model_name}_{semantic_type}_{epoch:04}.pt")

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': OPTIMISER.state_dict(),
                'loss': avg_train_loss,
            }
            torch.save(save_dict, model_path)


# Finish statement
print("Finished.")
