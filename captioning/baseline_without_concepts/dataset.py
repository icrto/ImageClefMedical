import os
import pandas as pd
import json
import torch
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def compute_stats(jsonfile, tokenizer):
    data = COCO(jsonfile)
    hist = []
    total = 0
    count = 0
    for item in data.getImgIds():
        id_cap = data.getAnnIds([item])
        captions = data.loadAnns(id_cap)
        for c in captions:
            caption = c["caption"]
            if len(caption) <= 0:
                print(caption, item)
                count += 1
            tokens = tokenizer(caption)
            hist.append(len(tokens.input_ids))
            total += len(tokens.input_ids)

    print(count)
    avg = total / len(hist)
    print(max(hist), min(hist), len(hist), avg)
    arr = np.array(hist)

    _, ax = plt.subplots()
    bins = [0, 30, 50, 100, 150, 200, 300, 400, 500, 600]
    _, _, bars = ax.hist(hist, bins=bins, histtype='bar', ec='black')
    ax.bar_label(bars)
    ax.set_xticks(bins)
    ax.set_xlabel('Caption length')
    ax.set_ylabel('Number of images')
    ax.axvline(arr.mean(), color='k', linestyle='dashed', linewidth=1)

    _, max_ylim = plt.ylim()
    ax.text(arr.mean() * 1.1, max_ylim * 0.9,
            'Mean: {:.2f}'.format(arr.mean()))

    dirname = os.path.dirname(jsonfile)
    if 'train' in jsonfile:
        plt.savefig(os.path.join(dirname, "caption_hist_train.png"))
    elif 'valid' in jsonfile:
        plt.savefig(os.path.join(dirname, "caption_hist_valid.png"))


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        json_file,
        tokenizer,
        max_length,
        feature_extractor,
        teacher_forcing=True,
        transform=None,
    ):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.json = json_file
        self.data = COCO(self.json)
        self.ids = self.data.getImgIds()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = feature_extractor
        self.teacher_forcing = teacher_forcing
        self.transform = transform

    def __getitem__(self, index):
        sample = {}

        idx = str(self.ids[index])

        # image
        img_data = self.data.imgs[idx]
        img_path = os.path.join(
            self.base_path,
            img_data["id"] + '.jpg',
        )

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        img_inputs = self.feature_extractor(images=img, return_tensors="pt")
        sample["pixel_values"] = img_inputs.pixel_values.squeeze(0)

        # text
        if self.teacher_forcing:
            caption = self.data.imgToAnns[idx]
            caption = caption[0]["caption"]

            sample["labels"] = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            # because the text model automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
            # We have to make sure that the PAD token is ignored
            sample["labels"] = torch.where(
                sample["labels"] == self.tokenizer.pad_token_id, -
                100, sample["labels"]
            )
        else:
            sample["id"] = idx

        return sample

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    compute_stats(
        "dataset/caption_prediction_train_coco.json", tokenizer
    )
    compute_stats(
        "dataset/caption_prediction_valid_coco.json", tokenizer
    )
