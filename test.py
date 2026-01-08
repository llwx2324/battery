import pandas as pd

train = pd.read_parquet(r"D:\thesis\thesis\01_data\processed\dataset_train.parquet")
calib = pd.read_parquet(r"D:\thesis\thesis\01_data\processed\dataset_calib.parquet")
test  = pd.read_parquet(r"D:\thesis\thesis\01_data\processed\dataset_test.parquet")

def show(name, df):
    print("\n==", name, "==")
    print("rows:", len(df), "cities:", df["Code"].nunique())
    print("min Year-Month:", df[["Year","Month"]].min().to_dict(),
          "max Year-Month:", df[["Year","Month"]].max().to_dict())
    print("y missing %:", df["PEV_number"].isna().mean())

show("train", train)
show("calib", calib)
show("test", test)
