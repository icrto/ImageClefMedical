import torch
from torch.utils.data import DataLoader
from baseline import model
from dataset import ImageDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms
import sklearn.metrics

# Arguments
DATA_DIR = "."
TOP_K_CONCEPTS = 100
SUBSET = 'valid_all' # change to topk to generate csv only for subset of images with topk concepts
BASE_DIR = "/BARRACUDA8T/ImageCLEF2022/dataset_resized"
TRAIN_FE = True # change to False to freeze entire feature extraction backbone
IMG_SIZE = (224, 224)

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# intialize the model
model = model(pretrained=False, requires_grad=TRAIN_FE, nr_concepts=TOP_K_CONCEPTS).to(device)

# load the model checkpoint
checkpoint = torch.load('densenet_asl/model_densenet121_100_asl_best.pth')
best_loss_epoch = checkpoint["epoch"]
print(f"Epoch of best val loss {best_loss_epoch}")

# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

if SUBSET == 'valid_topk':
    test_csv = os.path.join(BASE_DIR, 'new_val_subset_top100.csv')
elif SUBSET == 'valid_all':
    test_csv = os.path.join(BASE_DIR, 'concept_detection_valid.csv')
elif SUBSET == 'train_topk':
    test_csv = os.path.join(BASE_DIR, 'new_train_subset_top100.csv')
elif SUBSET == 'train_all':
    test_csv = os.path.join(BASE_DIR, 'concept_detection_train.csv')
elif SUBSET == 'test':
    test_csv = os.path.join(BASE_DIR, 'test_images.csv')
else:
    raise ValueError("Please choose only <valid_topk>, <valid_all>, <train_topk>, <train_all> or <test> for the SUBSET argument.")

if TOP_K_CONCEPTS == 100:
    df_all_concepts = pd.read_csv(os.path.join(BASE_DIR, "new_top100_concepts.csv"), sep="\t")
else:    
    df_all_concepts = pd.read_csv(os.path.join(BASE_DIR, "concepts.csv"), sep="\t")

all_concepts = df_all_concepts["concept"].tolist()


# prepare the test dataset and dataloader
transform = transforms.Compose([
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
test_data = ImageDataset(
    test_csv, df_all_concepts=None, transform=transform
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
    if('label' in data):
        target = data['label']
        y_true.append(target[0].numpy())

    # get the predictions by passing the image through the model
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()

    indices = np.where(outputs.numpy()[0] >= 0.5)  # decision threshold = 0.5

    # Add the valid concepts
    predicted_concepts = ""
    for i in indices[0]:
        predicted_concepts += f"{all_concepts[i]};"

    eval_images.append(image_ids[counter])
    eval_concepts.append(predicted_concepts[:-1])

    if('label' in data):
        zero_array = np.zeros_like(target[0])
        for idx in indices:
            zero_array[idx] = 1

        y_pred.append(zero_array)

# Generate Evaluation CSV
# Create a dictionary to obtain DataFrame
eval_set = dict()
eval_set["ID"] = eval_images
eval_set["cuis"] = eval_concepts

# Save this into .CSV
evaluation_df = pd.DataFrame(data=eval_set)
if(SUBSET == 'test'):
    evaluation_df.to_csv(os.path.join(DATA_DIR, f"eval_results_{SUBSET}.csv"), sep="|", index=False, header=False)
else:
    evaluation_df.to_csv(os.path.join(DATA_DIR, f"eval_results_{SUBSET}.csv"), sep="\t", index=False)

if len(y_true) > 0:
    print(f"/////////// Evaluation Report ////////////")
    print(f"Exact Match Ratio: {sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):.4f}")
    print(f"Hamming loss: {sklearn.metrics.hamming_loss(y_true, y_pred):.4f}")
    print(f"Recall: {sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
    print(f"Precision: {sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
    print(f"F1 Measure: {sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")