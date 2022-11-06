from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to run the script.")

    parser.add_argument(
        "--ann",
        type=str,
        default="dataset/caption_prediction_valid_coco.json",
        help="Annotation file to load.",
    )

    parser.add_argument(
        "res", type=str, help="Results file to load.",
    )

    args = parser.parse_args()

    save_path = os.path.dirname(args.res)

    # create coco object and cocoRes object
    coco_gts = COCO(args.ann)
    coco_res = coco_gts.loadRes(args.res)
    cocoEval = COCOEvalCap(coco_gts, coco_res)
    cocoEval.params["image_id"] = coco_res.getImgIds()
    cocoEval.evaluate()

    df = pd.DataFrame(
        columns=[
            "img_id",
            "generated",
            "original",
            "BLEU-1",
            "BLEU-2",
            "BLEU-3",
            "BLEU-4",
            "METEOR",
            "ROUGE_L",
            "CIDEr",
            "SPICE",
        ]
    )

    meteor = []
    for item in tqdm(cocoEval.evalImgs, total=len(cocoEval.evalImgs)):
        # get bleu, rouge, cider and spice from cocoeval
        img_id = item["image_id"]
        coco_id_cap = coco_res.getAnnIds([img_id])
        cap = coco_res.loadAnns(coco_id_cap)
        gen = cap[0]["caption"]

        coco_id_ref = coco_gts.getAnnIds([img_id])
        ref = coco_gts.loadAnns(coco_id_ref)
        refs = []
        for c in ref:
            refs.append(c["caption"])

        df = df.append(
            {
                "img_id": img_id,
                "generated": gen,
                "original": refs,
                "BLEU-1": np.round(item["Bleu_1"], 4),
                "BLEU-2": np.round(item["Bleu_2"], 4),
                "BLEU-3": np.round(item["Bleu_3"], 4),
                "BLEU-4": np.round(item["Bleu_4"], 4),
                "METEOR": np.round(item["METEOR"], 4),
                "ROUGE_L": np.round(item["ROUGE_L"], 4),
                "CIDEr": np.round(item["CIDEr"], 4),
                "SPICE": np.round(item["SPICE"]["All"]["f"], 4),
            },
            ignore_index=True,
        )

    # save captions and their individual metrics
    file_name = args.res.split("/")[-1].split("_")[0]
    df = df.sort_values(by=["img_id"])
    df.to_excel(os.path.join(save_path, file_name + "_reports.xlsx"))

    # save evaluation scores
    with open(os.path.join(save_path, file_name + "_metrics.txt"), "w") as f:
        for metric, score in cocoEval.eval.items():
            print(f"{metric}: {score:.3f}", file=f)
