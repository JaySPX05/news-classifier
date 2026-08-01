# models/compare.py
import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

DATA_DIR  = "D:/news-classifier/data"
MODEL_DIR = "D:/news-classifier/models"

LABEL2ID = {"center": 0, "left": 1, "right": 2}
ID2LABEL  = {0: "center", 1: "left", 2: "right"}

test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
X_test  = test_df["headline"].tolist()
y_test  = test_df["bias_label"].tolist()

# --- Baseline ---
with open(f"{MODEL_DIR}/baseline_vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)
with open(f"{MODEL_DIR}/baseline_model.pkl", "rb") as f:
    clf = pickle.load(f)
baseline_preds = clf.predict(vec.transform(X_test))
baseline_acc   = accuracy_score(y_test, baseline_preds)
baseline_f1    = f1_score(y_test, baseline_preds, average="macro")

# --- DistilBERT ---
tokenizer = DistilBertTokenizer.from_pretrained(f"{MODEL_DIR}/distilbert")
model     = DistilBertForSequenceClassification.from_pretrained(f"{MODEL_DIR}/distilbert")
model.eval()

inputs = tokenizer(X_test, truncation=True, padding=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
bert_preds = [ID2LABEL[i] for i in torch.argmax(logits, dim=1).tolist()]
bert_acc   = accuracy_score(y_test, bert_preds)
bert_f1    = f1_score(y_test, bert_preds, average="macro")

# --- Print comparison ---
print("=" * 45)
print(f"{'Model':<20} {'Accuracy':>10} {'Macro F1':>10}")
print("=" * 45)
print(f"{'Baseline (TF-IDF)':<20} {baseline_acc:>10.2%} {baseline_f1:>10.2%}")
print(f"{'DistilBERT':<20} {bert_acc:>10.2%} {bert_f1:>10.2%}")
print("=" * 45)

print("\nBaseline classification report:")
print(classification_report(y_test, baseline_preds, target_names=["center","left","right"]))

print("DistilBERT classification report:")
print(classification_report(y_test, bert_preds, target_names=["center","left","right"]))