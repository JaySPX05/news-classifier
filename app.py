import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import torch
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine

st.set_page_config(page_title="VERITAS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

#MainMenu, header, footer { display: none !important; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; background: #F2EFE6; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #F2EFE6 !important;
  border-right: 1px solid #C8C2AE !important;
  min-width: 200px !important;
  max-width: 200px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* Radio nav */
div[data-testid="stRadio"] label {
  font-family: 'Syne', sans-serif !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  color: #0A0A08 !important;
  padding: 8px 14px !important;
  display: flex !important;
  align-items: center !important;
  border-bottom: 1px solid #C8C2AE !important;
  cursor: pointer !important;
  transition: background 0.1s !important;
  letter-spacing: 0.02em !important;
}

div[data-testid="stRadio"] label:hover { background: #E8E4D8 !important; }
div[data-testid="stRadio"] input:checked + div { font-weight: 700 !important; }
div[data-testid="stRadio"] > div { gap: 0 !important; }
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  font-family: 'Syne', sans-serif !important;
  font-size: 12px !important;
}
/* Hide radio circles */
div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }
div[data-testid="stRadio"] > label:first-child { display: none !important; }
div[data-testid="stRadio"] p { font-family: 'Syne', sans-serif !important; font-size: 12px !important; }
div[data-testid="stRadio"] > div > label > div > p { font-family: 'Syne', sans-serif !important; }
div[data-testid="stRadio"] p { color: #0A0A08 !important; }
div[data-testid="stRadio"] span { color: #0A0A08 !important; }
div[data-testid="stRadio"] div { color: #0A0A08 !important; }
.stRadio > label { display: none !important; }

/* Text inputs */
.stTextInput input {
  font-family: 'DM Serif Display', Georgia, serif !important;
  font-size: 15px !important;
  background: #F2EFE6 !important;
  border: none !important;
  border-bottom: 2px solid #0A0A08 !important;
  border-radius: 0 !important;
  color: #0A0A08 !important;
  padding: 8px 4px !important;
  box-shadow: none !important;
}
.stTextInput input:focus {
  border-bottom-color: #C41E2A !important;
  box-shadow: none !important;
}
.stTextInput input::placeholder { color: #9E9780 !important; font-style: italic !important; }
.stTextInput label { display: none !important; }

/* Buttons */
.stButton > button {
  font-family: 'Syne', sans-serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  background: #0A0A08 !important;
  color: #F2EFE6 !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 8px 20px !important;
  width: 100% !important;
}
.stButton > button:hover { background: #C41E2A !important; border: none !important; }
.stButton > button:focus { box-shadow: none !important; border: none !important; }

/* Remove streamlit decorations */
.stTextInput > div > div { border: none !important; box-shadow: none !important; }
div[data-baseweb="input"] { background: transparent !important; }

/* Page background */
body { background: #F2EFE6 !important; }
.main { background: #F2EFE6 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bias_model():
    tokenizer = DistilBertTokenizer.from_pretrained("JaySPS/news-bias-classifier")
    model = DistilBertForSequenceClassification.from_pretrained("JaySPS/news-bias-classifier")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_fake_model():
    tokenizer = DistilBertTokenizer.from_pretrained("JaySPS/fake-news-detector")
    model = DistilBertForSequenceClassification.from_pretrained("JaySPS/fake-news-detector")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_data(ttl=3600)
def load_data():
    import os
    db_url = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
    # Fix URL format if needed
    db_url = db_url.replace("postgresql+psycopg2://", "postgresql://")
    engine = create_engine(db_url)
    df = pd.read_sql(
        "SELECT headline, source, bias_label, sentiment, scraped_at FROM headlines WHERE bias_label IS NOT NULL",
        engine
    )
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    df["date"] = df["scraped_at"].dt.date.astype(str)
    return df

def predict_bias(headline, tokenizer, model, analyzer):
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    id2label = {0: "center", 1: "left", 2: "right"}
    label_id = torch.argmax(logits, dim=1).item()
    sentiment = analyzer.polarity_scores(headline)["compound"]
    return {
        "bias": id2label[label_id],
        "confidence": round(max(probs), 4),
        "sentiment": round(sentiment, 4),
        "probs": {"center": round(probs[0], 4), "left": round(probs[1], 4), "right": round(probs[2], 4)}
    }

def predict_fake(headline, tokenizer, model):
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    label_id = torch.argmax(logits, dim=1).item()
    return {
        "label": {0: "fake", 1: "real"}[label_id],
        "confidence": round(max(probs), 4),
        "probs": {"fake": round(probs[0], 4), "real": round(probs[1], 4)}
    }

# Load
df = load_data()
bias_tokenizer, bias_model = load_bias_model()
fake_tokenizer, fake_model = load_fake_model()
analyzer = load_analyzer()

# Session state
for key, default in [
    ("bias_pred", {}), ("fake_pred", {}), ("bias_pred_a", {}),
    ("query", ""), ("qa", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Prepare data
bias_counts = df["bias_label"].value_counts().to_dict()
sentiment_by_source = df.groupby("source")["sentiment"].mean().round(3).to_dict()
daily = df.groupby(["date", "source"])["sentiment"].mean().round(3).reset_index()
timeline_data = {}
for source in df["source"].unique():
    src_data = daily[daily["source"] == source]
    timeline_data[source] = {"dates": src_data["date"].tolist(), "values": src_data["sentiment"].tolist()}

recent_headlines = df.sort_values("scraped_at", ascending=False).head(50)[
    ["headline", "source", "bias_label", "sentiment"]
].to_dict(orient="records")

# ── Sidebar navigation ──
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 14px;border-bottom:2px solid #0A0A08;margin-bottom:0">
      <div style="font-family:'DM Serif Display',serif;font-size:22px;color:#0A0A08;letter-spacing:-0.01em">VERI<span style="font-style:italic;color:#C41E2A">TAS</span></div>
      <div style="font-family:'DM Mono',monospace;font-size:8px;color:#7A7660;letter-spacing:0.15em;text-transform:uppercase;margin-top:2px">News Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)

    tab = st.radio("nav", [
        "01  Overview",
        "02  Bias Analysis",
        "03  Compare",
        "04  Fake Detector"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top:auto;padding:12px 14px;border-top:1px solid #C8C2AE;position:fixed;bottom:0;width:198px;background:#F2EFE6">
      <div style="font-family:'DM Mono',monospace;font-size:8px;color:#7A7660;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">Models Loaded</div>
      <div style="font-family:'DM Mono',monospace;font-size:9px;background:#0A0A08;color:#F2EFE6;padding:2px 6px;display:inline-block;margin-bottom:6px">DISTILBERT-BASE</div>
      <div style="font-family:'DM Mono',monospace;font-size:8px;color:#7A7660;margin-bottom:2px">BIAS · 77.4% ACC</div>
      <div style="height:2px;background:#C8C2AE;margin-bottom:6px"><div style="height:100%;width:77.4%;background:#0A0A08"></div></div>
      <div style="font-family:'DM Mono',monospace;font-size:8px;color:#7A7660;margin-bottom:2px">FAKE · 98.7% ACC</div>
      <div style="height:2px;background:#C8C2AE"><div style="height:100%;width:98.7%;background:#0A0A08"></div></div>
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──
active = tab.split("  ")[1].lower().replace(" ", "")

def render_results(mode, bias_pred, fake_pred, bias_pred_a, query, qa):
    with open("results.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__MODE__",      mode)
    html = html.replace("__BIAS_PRED__", json.dumps(bias_pred))
    html = html.replace("__FAKE_PRED__", json.dumps(fake_pred))
    html = html.replace("__BIAS_PRED_A__", json.dumps(bias_pred_a))
    html = html.replace('"__QUERY__"',   json.dumps(query))
    html = html.replace('"__QA__"',      json.dumps(qa))
    components.html(html, height=500, scrolling=False)

if active == "overview":
    with open("dashboard.html", "r", encoding="utf-8") as f:
        html = f.read()
    app_data = {
        "total": len(df), "sources": int(df["source"].nunique()),
        "bias_counts": bias_counts, "sentiment_by_source": sentiment_by_source,
        "timeline": timeline_data, "headlines": recent_headlines,
        "active_tab": "dashboard", "query": "", "qa": "",
    }
    html = html.replace("__APP_DATA__",    json.dumps(app_data))
    html = html.replace("__BIAS_PRED__",   json.dumps({}))
    html = html.replace("__FAKE_PRED__",   json.dumps({}))
    html = html.replace("__BIAS_PRED_A__", json.dumps({}))
    components.html(html, height=900, scrolling=True)

elif active == "biasanalysis":
    st.markdown("""
    <div style="font-family:'DM Serif Display',serif;font-size:22px;color:#0A0A08;border-bottom:2px solid #0A0A08;padding-bottom:6px;margin-bottom:16px">
      Bias Analysis <span style="font-family:'DM Mono',monospace;font-size:10px;color:#7A7660;font-style:normal">· DistilBERT · 77.4% accuracy</span>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        cls_q = st.text_input("h", value=st.session_state.query,
            placeholder="Paste or type a news headline...", key="cls_q")
    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("▶ Analyze", key="cls_btn"):
            if cls_q:
                st.session_state.query = cls_q
                st.session_state.bias_pred = predict_bias(cls_q, bias_tokenizer, bias_model, analyzer)
                st.session_state.fake_pred = {}
                st.session_state.bias_pred_a = {}
                st.rerun()
    render_results("classify", st.session_state.bias_pred, {}, {}, st.session_state.query, "")

elif active == "compare":
    st.markdown("""
    <div style="font-family:'DM Serif Display',serif;font-size:22px;color:#0A0A08;border-bottom:2px solid #0A0A08;padding-bottom:6px;margin-bottom:16px">
      Comparative Analysis <span style="font-family:'DM Mono',monospace;font-size:10px;color:#7A7660;font-style:normal">· side-by-side</span>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:8px;color:#7A7660;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:4px'>Headline A</div>", unsafe_allow_html=True)
        cmp_a = st.text_input("a", value=st.session_state.qa,
            placeholder="First headline...", key="cmp_a")
    with col2:
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:8px;color:#7A7660;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:4px'>Headline B</div>", unsafe_allow_html=True)
        cmp_b = st.text_input("b", value=st.session_state.query,
            placeholder="Second headline...", key="cmp_b")
    col3, _ = st.columns([1, 4])
    with col3:
        if st.button("▶ Compare", key="cmp_btn"):
            if cmp_a and cmp_b:
                st.session_state.query = cmp_b
                st.session_state.qa = cmp_a
                st.session_state.bias_pred = predict_bias(cmp_b, bias_tokenizer, bias_model, analyzer)
                st.session_state.bias_pred_a = predict_bias(cmp_a, bias_tokenizer, bias_model, analyzer)
                st.session_state.fake_pred = {}
                st.rerun()
    render_results("compare", st.session_state.bias_pred, {}, st.session_state.bias_pred_a, st.session_state.query, st.session_state.qa)

elif active == "fakedetector":
    st.markdown("""
    <div style="font-family:'DM Serif Display',serif;font-size:22px;color:#0A0A08;border-bottom:2px solid #0A0A08;padding-bottom:6px;margin-bottom:16px">
      Authenticity Verification <span style="font-family:'DM Mono',monospace;font-size:10px;color:#7A7660;font-style:normal">· 98.7% accuracy</span>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        fake_q = st.text_input("f", value=st.session_state.query,
            placeholder="Paste a headline to verify its authenticity...", key="fake_q")
    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("▶ Verify", key="fake_btn"):
            if fake_q:
                st.session_state.query = fake_q
                st.session_state.bias_pred = predict_bias(fake_q, bias_tokenizer, bias_model, analyzer)
                st.session_state.fake_pred = predict_fake(fake_q, fake_tokenizer, fake_model)
                st.session_state.bias_pred_a = {}
                st.rerun()
    render_results("fakenews", st.session_state.bias_pred, st.session_state.fake_pred, {}, st.session_state.query, "")