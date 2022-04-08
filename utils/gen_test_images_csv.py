import os
import pandas as pd

data_dir = "/BARRACUDA8T/ImageCLEF2022/dataset/test"
df = pd.DataFrame(columns=["ID"])

df["ID"] = os.listdir(data_dir)
df["ID"] = df["ID"].apply(lambda x: x[:-4])

df.to_csv(os.path.join(os.path.dirname(data_dir), "test_images.csv"), index=False)
print(df)