import sys
import os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
    import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from baseline import build_model, do_epoch
from dataset import ImageDataset
from asl import AsymmetricLoss
from preprocessing.aux_functions import get_class_weights
import numpy as np
import argparse
import datetime

def _create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(folder, timestamp)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return timestamp, results_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to run the script.")

    # Processing parameters
    parser.add_argument("--gpu_id",
                        type=str,
                        default="0")
    parser.add_argument("--num_workers",
                        type=int,
                        default=4,
                        help="Number of workers for dataloader.")

    # Directories and paths
    parser.add_argument("--basedir",
                        type=str,
                        default=
                        "dataset",
                        help="Directory where dataset is stored.")
    parser.add_argument("--logdir",
                        type=str,
                        default=
                        "results",
                        help="Directory where logs and models are to be stored." )

    # Model
    parser.add_argument("--nr_concepts",
                        type=str,
                        default='all',
                        help="Number of concepts to predict. Example: all, 100, etc.")
    parser.add_argument("--freeze_fe",
                        action="store_true",
                        help="Freeze model (feature extractor) backbone.")

    # Training
    parser.add_argument("--loss_fn",
                        type=str,
                        default='bce',
                        choices=['bce', 'bce_weighted', 'asl'],
                        help="Loss function to use.")
    parser.add_argument("--ckpt",
                        type=str,
                        default=None,
                        help="Load model from this checkpoint.")
    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume training from checkpoint.")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="Number of epochs.")
    parser.add_argument("--bs",
                        type=int,
                        default=32,
                        help="Batch size.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Learning rate." )

    args = parser.parse_args()

    if args.resume:
        assert args.ckpt is not None, "Please specify the model checkpoint when resume is True"

    # initialize the computation device
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    timestamp, save_path = _create_folder(args.logdir)
    with open(os.path.join(save_path, "train_params.txt"), "w") as f:
        f.write(str(args))
        
    tb_writer = SummaryWriter(log_dir=save_path)

    # arguments
    BASE_DIR = args.basedir
    NR_CONCEPTS = args.nr_concepts
    FREEZE_FE = args.freeze_fe
    WORKERS = args.num_workers
    IMG_SIZE = (224, 224)
    LOSS = args.loss_fn
    LR = args.lr
    EPOCHS = args.epochs
    BS = args.bs

    #intialize the model
    model = build_model(pretrained=True, freeze_fe=FREEZE_FE, nr_concepts=NR_CONCEPTS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # read the training csv file
    if NR_CONCEPTS == 'all':
        train_csv = os.path.join(BASE_DIR, 'concept_detection_train.csv')
        val_csv = os.path.join(BASE_DIR, 'concept_detection_valid.csv')
        concept_dict = os.path.join(BASE_DIR, "concepts.csv")
    else:
        train_csv = os.path.join(BASE_DIR, f'concept_detection_train_top{NR_CONCEPTS}.csv')
        val_csv = os.path.join(BASE_DIR, f'concept_detection_valid_top{NR_CONCEPTS}.csv')
        concept_dict = os.path.join(BASE_DIR, f'concepts_top{NR_CONCEPTS}.csv')

    # train dataset
    train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(IMG_SIZE),
                    transforms.RandomRotation(degrees=45),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])
    train_data = ImageDataset(
        train_csv, df_all_concepts=concept_dict, transform=train_transform
    )

    # validation dataset
    val_transform = transforms.Compose([
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])
    valid_data = ImageDataset(
        val_csv, df_all_concepts=concept_dict, transform=val_transform
    )

    # train data loader
    train_loader = DataLoader(
        train_data,
        batch_size=BS,
        shuffle=True,
        num_workers=WORKERS
    )

    # validation data loader
    valid_loader = DataLoader(
        valid_data,
        batch_size=BS,
        shuffle=False,
        num_workers=WORKERS

    )

    weights = None
    if 'bce' in LOSS:
        criterion = nn.BCELoss()
        if LOSS == 'bce_weighted':
            weights_numpy = get_class_weights(dataset_csv=train_csv, concept_csv=concept_dict, n_concepts=NR_CONCEPTS)
            weights = torch.from_numpy(weights_numpy).to(device)
    elif (LOSS == "asl"):
        criterion = AsymmetricLoss()

    # resume training from given checkpoint
    if args.resume:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from {args.ckpt} at epoch {init_epoch}")
    else:
        init_epoch = 0

    # start the training and validation
    train_loss = []
    valid_loss = []
    best_val_loss = np.inf
    for epoch in range(init_epoch, EPOCHS):
        print(f"Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss = do_epoch(
            model, train_loader, criterion, device, optimizer, weights=weights
        )
        valid_epoch_loss = do_epoch(
            model, valid_loader, criterion, device, weights=weights, validation=True
        )
        if valid_epoch_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss} to {valid_epoch_loss}")
            best_val_loss = valid_epoch_loss
            # save model with best val loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_epoch_loss,
            }, os.path.join(save_path, 'best_model.pt'))

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {valid_epoch_loss:.4f}')
        tb_writer.add_scalar('loss/train', train_epoch_loss, epoch)
        tb_writer.add_scalar('loss/val', valid_epoch_loss, epoch)

    # save the trained model to disk
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_epoch_loss,
    }, os.path.join(save_path, 'last_model.pt'))
