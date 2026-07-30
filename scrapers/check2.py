import pandas as pd

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"D:/news-classifier/data/{split}.csv")
    print(split + ".csv - " + str(len(df)) + " rows")
    print(df["bias_label"].value_counts().to_string())
    print("Sample: " + df["headline"].iloc[0])
    print("")