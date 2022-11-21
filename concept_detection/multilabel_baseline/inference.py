import torch
from torch.utils.data import DataLoader
from baseline import build_model
from dataset import ImageDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms
import sklearn.metrics
import argparse

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

    # Model
    parser.add_argument("--nr_concepts",
                        type=str,
                        default='all',
                        help="Number of concepts the model was trained with. Example: all, 100, etc.")
    parser.add_argument("--ckpt",
                        type=str,
                        required=True,
                        help="Load model from this checkpoint.")
    
    # Eval
    parser.add_argument("--images_csv",
                    type=str,
                    required=True,
                    help="csv with the images on which to run inference and generate predictions.")

    args = parser.parse_args()

    BASE_DIR = args.basedir
    LOGDIR = os.path.dirname(args.ckpt) # results are saved in model checkpoint's folder
    NR_CONCEPTS = args.nr_concepts
    if NR_CONCEPTS == 'all': NR_CONCEPTS = 8374
    else: NR_CONCEPTS = int(NR_CONCEPTS)
    CKPT = args.ckpt
    IMG_SIZE = (224, 224)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # intialize the model
    model = build_model(pretrained=False, nr_concepts=NR_CONCEPTS).to(device)

    # load the model checkpoint
    checkpoint = torch.load(CKPT)
    best_loss_epoch = checkpoint["epoch"]
    print(f"Epoch of best val loss {best_loss_epoch}")

    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if NR_CONCEPTS == 8374:
        df_all_concepts = pd.read_csv(os.path.join(BASE_DIR, "concepts.csv"), sep="\t")
    else:
        df_all_concepts = pd.read_csv(os.path.join(BASE_DIR, f'concepts_top{NR_CONCEPTS}.csv'), sep="\t")

    concepts_mapping = df_all_concepts["concept"].tolist()

    # prepare the test dataset and dataloader
    transform = transforms.Compose([
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])
    test_data = ImageDataset(
        args.images_csv, df_all_concepts=None, transform=transform
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    image_ids = test_data.csv["ID"]

    y_true = []
    y_pred = []
    eval_images = []
    eval_concepts = []
    for counter, data in enumerate(tqdm(test_loader)):
        image = data['image'].to(device)

        # get the predictions by passing the image through the model
        outputs = model(image)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu()

        indices = np.where(outputs.numpy()[0] >= 0.5)  # decision threshold = 0.5

        # add the valid concepts
        predicted_concepts = ""
        for i in indices[0]:
            predicted_concepts += f"{concepts_mapping[i]};"

        eval_images.append(image_ids[counter])
        eval_concepts.append(predicted_concepts[:-1])

    # generate Evaluation CSV
    # create a dictionary to obtain DataFrame
    eval_set = dict()
    eval_set["ID"] = eval_images
    eval_set["cuis"] = eval_concepts

    # Save this into .CSV
    evaluation_df = pd.DataFrame(data=eval_set)
    if('test' in args.images_csv):
        evaluation_df.to_csv(os.path.join(LOGDIR, f"preds_test.csv"), sep="|", index=False, header=False)
    else:
        subset = args.images_csv.split('_')
        subset = "_".join(subset[1:])
        subset = subset[:-4]
        evaluation_df.to_csv(os.path.join(LOGDIR, f"preds_{subset}.csv"), sep="|", index=False, header=False)