import pandas as pd
import json
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument(
    "--submission",
    action='store_true',
    help="Generate test submission.")
parser.add_argument("file",
                    type=str,
                    help="Prediction file to convert.")
args = parser.parse_args()

with open(args.file, 'r') as f:
    res = json.load(f)

df = pd.DataFrame(res)
df = df.rename(columns={"image_id": "ID"})

save_path = os.path.dirname(args.file)

if(args.submission):
    df.to_csv(os.path.join(save_path, "test_submission.csv"),
              sep='|', index=False, header=None)
else:
    df.to_csv(os.path.join(save_path, "val_preds.csv"), sep='\t', index=False)
