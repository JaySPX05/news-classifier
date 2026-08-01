import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import os

DB_PATH = "D:/news-classifier/scrapers/data/headlines.db"
OUT_DIR = "D:/news-classifier/data"

def export():
    os.makedirs(OUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
    "SELECT headline, source, bias_label, sentiment FROM headlines WHERE bias_label IS NOT NULL",
    conn
)
    conn.close()

    print(f"Total labeled headlines: {len(df)}")
    print("Label distribution:")
    print(df["bias_label"].value_counts().to_string())

    train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["bias_label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["bias_label"])

    train.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("Exported:")
    print(f"  train.csv  {len(train)} rows")
    print(f"  val.csv    {len(val)} rows")
    print(f"  test.csv   {len(test)} rows")

if __name__ == "__main__":
    export()