import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

DATA_DIR  = "D:/news-classifier/data"
MODEL_DIR = "D:/news-classifier/models/distilbert"

LABEL2ID = {"center": 0, "left": 1, "right": 2}
ID2LABEL = {0: "center", 1: "left", 2: "right"}

def load_data(path):
    df = pd.read_csv(path)
    df["label"] = df["bias_label"].map(LABEL2ID)
    return df

def tokenize(batch, tokenizer):
    return tokenizer(
        batch["headline"],
        truncation=True,
        padding="max_length",
        max_length=64      # headlines are short, 64 tokens is enough
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

def train():
    print("Loading data...")
    train_df = load_data(f"{DATA_DIR}/train.csv")
    val_df   = load_data(f"{DATA_DIR}/val.csv")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = Dataset.from_pandas(train_df[["headline", "label"]])
    val_ds = Dataset.from_pandas(val_df[["headline", "label"]])

    print("Tokenizing...")
    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda b: tokenize(b, tokenizer), batched=True)

    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=5,         # 5 passes over the data
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        learning_rate=2e-5,         # standard for fine-tuning BERT
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Training... (this will take 20-30 mins on CPU)")
    trainer.train()

    print("Saving model...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("Evaluating on validation set...")
    preds_out = trainer.predict(val_ds)
    preds = np.argmax(preds_out.predictions, axis=1)
    labels = val_ds["label"]
    print(classification_report(labels, preds, target_names=["center","left","right"]))

if __name__ == "__main__":
    train()