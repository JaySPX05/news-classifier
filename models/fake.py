# models/fake_news.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch
import pickle
import os

DATA_DIR  = "D:/news-classifier/data"
MODEL_DIR = "D:/news-classifier/models/fake_news_distilbert"

def load_data():
    real = pd.read_csv(f"{DATA_DIR}/True.csv")
    fake = pd.read_csv(f"{DATA_DIR}/Fake.csv")

    real["label"] = 1  # 1 = real
    fake["label"] = 0  # 0 = fake

    df = pd.concat([real, fake], ignore_index=True)
    df = df[["title", "label"]].dropna()
    df = df.rename(columns={"title": "headline"})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"Total: {len(df)} headlines")
    print(f"Real: {len(df[df['label']==1])}, Fake: {len(df[df['label']==0])}")
    return df

def train_baseline(train_df, val_df):
    print("\nTraining baseline...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X_train = vec.fit_transform(train_df["headline"])
    X_val   = vec.transform(val_df["headline"])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_df["label"])

    preds = clf.predict(X_val)
    acc = accuracy_score(val_df["label"], preds)
    print(f"Baseline accuracy: {acc:.2%}")
    print(classification_report(val_df["label"], preds, target_names=["fake","real"]))

    os.makedirs("D:/news-classifier/models", exist_ok=True)
    with open("D:/news-classifier/models/fakenews_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open("D:/news-classifier/models/fakenews_baseline.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Baseline saved.")

def tokenize(batch, tokenizer):
    return tokenizer(batch["headline"], truncation=True, padding="max_length", max_length=64)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

def train_distilbert(train_df, val_df):
    print("\nFine-tuning DistilBERT for fake news detection...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = Dataset.from_pandas(train_df[["headline","label"]].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[["headline","label"]].reset_index(drop=True))

    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda b: tokenize(b, tokenizer), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "fake", 1: "real"},
        label2id={"fake": 0, "real": 1}
    )

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Training... (this will take 30-60 mins on CPU)")
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("DistilBERT fake news model saved.")

    preds_out = trainer.predict(val_ds)
    preds = np.argmax(preds_out.predictions, axis=1)
    print(classification_report(val_ds["label"], preds, target_names=["fake","real"]))

def main():
    df = load_data()
    train_df, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_baseline(train_df, val_df)
    train_distilbert(train_df, val_df)

if __name__ == "__main__":
    main()