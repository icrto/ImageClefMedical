from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    VisionEncoderDecoderModel
)
import argparse
import os
from dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
from tqdm import tqdm
import json
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to run the script.")

    # Processing parameters
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Which gpus to use in CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for dataloader."
    )

    # Directories and paths
    parser.add_argument(
        "ckpt",
        type=str,
        help="Model to be loaded.",
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=128,
        help="Batch size.",
    )


    args = parser.parse_args()
    if args.ckpt[-1] == "/":
        args.ckpt = args.ckpt[:-1]
    save_path = os.path.dirname(args.ckpt)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    model = VisionEncoderDecoderModel.from_pretrained(
        args.ckpt
    )

    model.eval()
    model.to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ckpt)
    size = feature_extractor.size
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

    # Data
    preprocess = T.Compose([T.CenterCrop(size), T.ToTensor()])
    val_dtset = Dataset(
        "dataset/captions/valid",
        "dataset/captions/caption_prediction_valid_coco.json",
        tokenizer,
        512,
        feature_extractor,
        transform=preprocess,
        teacher_forcing=False,
    )
    val_loader = DataLoader(
        val_dtset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )


    with torch.no_grad():
        res_coco = []
        res_clef = []

        for sample in tqdm(val_loader):
            ids = sample['id']
            names = sample['image_name']
            sample.pop('id')
            sample.pop('image_name')

            sample = {k: v.to(device) for k, v in sample.items()}
            output = model.generate(**sample)

            pred_str = tokenizer.batch_decode(output, skip_special_tokens=True)

            for i in range(len(ids)):
                res_coco.append(
                    {
                        "image_id": ids[i].item(),
                        "caption": pred_str[i],
                    }
                )
                res_clef.append({
                    "ID": names[i],
                    "caption": pred_str[i],
                })


        with open(os.path.join(save_path, "val_preds.json"), "w") as f:
            json.dump(res_coco, f)

        df = pd.DataFrame(res_clef)
        df.to_csv(os.path.join(save_path, "val_preds.csv"), sep='\t', index=False)
