# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DB_PATH   = "D:/news-classifier/scrapers/data/headlines.db"
MODEL_DIR = "D:/news-classifier/models/distilbert"

ID2LABEL = {0: "center", 1: "left", 2: "right"}
LABEL_COLORS = {"left": "#4A90D9", "center": "#7B7B7B", "right": "#E05C5C"}

# ── Load model once (cached so it doesn't reload on every interaction) ──
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

# ── Load data from DB ──
@st.cache_data(ttl=3600)  # refresh every hour
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT headline, source, bias_label, sentiment, scraped_at FROM headlines WHERE bias_label IS NOT NULL",
        conn
    )
    conn.close()
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    df["date"] = df["scraped_at"].dt.date
    return df

def predict(headline, tokenizer, model, analyzer):
    # Bias prediction
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    label_id = torch.argmax(logits, dim=1).item()
    bias = ID2LABEL[label_id]
    confidence = torch.softmax(logits, dim=1).max().item()

    # Sentiment
    sentiment = analyzer.polarity_scores(headline)["compound"]

    return bias, confidence, sentiment

# ── App layout ──
st.set_page_config(page_title="News Bias Classifier", layout="wide")
st.title("News Headline Bias Classifier")
st.caption("Classifying political bias and sentiment across major news sources")

df = load_data()
tokenizer, model = load_model()
analyzer = load_analyzer()

# ── Row 1: KPI metrics ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Headlines", len(df))
col2.metric("Sources", df["source"].nunique())
col3.metric("Left Headlines",   len(df[df["bias_label"] == "left"]))
col4.metric("Right Headlines",  len(df[df["bias_label"] == "right"]))

st.divider()

# ── Row 2: Bias distribution + Sentiment by source ──
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Bias Distribution")
    bias_counts = df["bias_label"].value_counts().reset_index()
    bias_counts.columns = ["label", "count"]
    fig1 = px.pie(
        bias_counts, values="count", names="label",
        color="label", color_discrete_map=LABEL_COLORS,
        hole=0.4
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Average Sentiment by Source")
    sent_by_source = df.groupby("source")["sentiment"].mean().reset_index()
    sent_by_source.columns = ["source", "avg_sentiment"]
    sent_by_source = sent_by_source.sort_values("avg_sentiment")
    fig2 = px.bar(
        sent_by_source, x="avg_sentiment", y="source",
        orientation="h", color="avg_sentiment",
        color_continuous_scale="RdYlGn", range_color=[-0.5, 0.5]
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 3: Sentiment timeline ──
st.subheader("Sentiment Over Time by Source")
daily = df.groupby(["date", "source"])["sentiment"].mean().reset_index()
fig3 = px.line(
    daily, x="date", y="sentiment", color="source",
    markers=True
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Row 4: Live classifier ──
st.subheader("Live Headline Classifier")
st.caption("Type any headline and get an instant bias and sentiment prediction")

headline_input = st.text_input("Enter a headline:", placeholder="e.g. Government announces new climate policy")

if headline_input:
    bias, confidence, sentiment = predict(headline_input, tokenizer, model, analyzer)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Bias", bias.upper())
    c2.metric("Confidence", f"{confidence:.1%}")

    sentiment_label = "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"
    c3.metric("Sentiment", f"{sentiment_label} ({sentiment:.2f})")

st.divider()

# ── Row 5: Recent headlines table ──
st.subheader("Recent Headlines")
show_source = st.selectbox("Filter by source", ["All"] + sorted(df["source"].unique().tolist()))
filtered = df if show_source == "All" else df[df["source"] == show_source]
st.dataframe(
    filtered[["headline", "source", "bias_label", "sentiment", "date"]]
    .sort_values("date", ascending=False)
    .head(50),
    use_container_width=True
)