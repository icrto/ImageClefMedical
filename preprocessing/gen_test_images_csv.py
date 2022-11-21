import os
import pandas as pd
import argparse
parser = argparse.ArgumentParser(
    description="Arguments to run the script.")

parser.add_argument("--datadir",
                    type=str, required=True)
args = parser.parse_args()

if args.datadir.endswith('/'):
    args.datadir = args.datadir[:-1]

df = pd.DataFrame(columns=["ID"])

df["ID"] = os.listdir(args.datadir)
df["ID"] = df["ID"].apply(lambda x: x[:-4])

savepath = os.path.join(os.path.dirname(args.datadir), "test_images.csv")
df.to_csv(savepath, index=False)
print("Saved dataframe to: ", savepath)