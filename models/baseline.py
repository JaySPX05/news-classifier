import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

DATA_DIR = "D:/news-classifier/data"
MODEL_DIR = "D:/news-classifier/models"

def train():
    # Load data
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    val_df   = pd.read_csv(f"{DATA_DIR}/val.csv")

    X_train = train_df["headline"]
    y_train = train_df["bias_label"]
    X_val   = val_df["headline"]
    y_val   = val_df["bias_label"]

    print(f"Training on {len(X_train)} headlines...")

    # Step 1: convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000,    # keep top 5000 words
        ngram_range=(1, 2),   # use single words AND pairs (bigrams)
        stop_words="english"  # remove common words like 'the', 'a'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)

    # Step 2: train logistic regression
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    # Step 3: evaluate
    val_preds = model.predict(X_val_vec)
    acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {acc:.2%}")
    print("Classification report:")
    print(classification_report(y_val, val_preds))

    # Step 4: save model to disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/baseline_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/baseline_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Model saved to models/baseline_model.pkl")

    # Step 5: show top predictive words per class
    print("Top words per class:")
    feature_names = vectorizer.get_feature_names_out()
    for i, class_name in enumerate(model.classes_):
        top_idx = model.coef_[i].argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        print(f"  {class_name}: {', '.join(top_words)}")

if __name__ == "__main__":
    train()