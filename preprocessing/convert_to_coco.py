import json
import os
import pandas as pd


def convert_coco_format(csv_file):
    path = os.path.dirname(csv_file)
    # training
    train_df = pd.read_csv(csv_file, sep='\t')

    train_dict = {"images": [], "annotations": []}
    for idx, row in train_df.iterrows():
        train_dict["images"].append({"id": row["ID"]})
        train_dict["annotations"].append(
            {"image_id": row["ID"], "id": idx, "caption": row["caption"]})

    new_filename = csv_file.split('/')[-1][:-4] + "_coco.json"
    with open(os.path.join(path, new_filename), 'w') as outfile:
        json.dump(train_dict, outfile)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="File to convert to COCO format.")
    args = parser.parse_args()

    convert_coco_format(args.file)
