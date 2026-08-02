# News Headline Bias Classifier

A end-to-end NLP pipeline that scrapes, classifies, and visualizes political bias and sentiment across major news sources. Live dashboard at [news-classifier-ecjkvdevhdoq9q5btaa2qp.streamlit.app](https://news-classifier-ecjkvdevhdoq9q5btaa2qp.streamlit.app)

---

## What it does

- Scrapes headlines daily from 5 news sources (BBC, NPR, TheGuardian, FoxNews, AlJazeera) via RSS feeds
- Classifies each headline as **left**, **center**, or **right** leaning using a fine-tuned DistilBERT model
- Scores emotional tone using VADER sentiment analysis
- Stores everything in a cloud PostgreSQL database (Supabase)
- Displays live charts and a real-time headline classifier on a public Streamlit dashboard

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (TF-IDF + Logistic Regression) | 47% | 0.38 |
| Fine-tuned DistilBERT | 77% | 0.76 |

DistilBERT achieved **64% relative improvement** in macro F1 over the baseline.

### Per-class performance (DistilBERT)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Center | 0.75 | 0.91 | 0.82 |
| Left | 1.00 | 0.53 | 0.69 |
| Right | 0.69 | 0.85 | 0.76 |

---

## Architecture
RSS Feeds (5 sources)
↓
scraper.py ← runs daily via Windows Task Scheduler
↓
Supabase (PostgreSQL) ← cloud database
↓
label.py ← weak supervision labeling by source
sentiment.py ← VADER sentiment scoring
↓
DistilBERT model ← fine-tuned on labeled headlines
(hosted on HuggingFace Hub: JaySPS/news-bias-classifier)
↓
app.py ← Streamlit dashboard (live public URL)
---

## Data

- **529 headlines** collected across 5 sources
- **Labeling strategy:** Weak supervision — labels assigned by source outlet based on AllSides Media Bias Ratings
- **Sources and labels:**
  - BBC → center
  - AlJazeera → center
  - NPR → left
  - TheGuardian → left
  - FoxNews → right
- **Split:** 80% train / 10% validation / 10% test

---

## Key findings

- All news sources have negative average sentiment — news is inherently negative
- TheGuardian has the most negative sentiment (-0.226), BBC the least (-0.079)
- FoxNews and BBC have almost identical sentiment scores despite opposite political leanings — sentiment alone does not predict bias
- DistilBERT achieved perfect precision (1.00) on left-leaning headlines

---

## Stack

| Component | Technology |
|---|---|
| Scraping | Python, requests, BeautifulSoup, RSS |
| Database | Supabase (PostgreSQL), SQLite (local) |
| NLP | HuggingFace Transformers, DistilBERT |
| Baseline ML | scikit-learn, TF-IDF, Logistic Regression |
| Sentiment | VADER (vaderSentiment) |
| Dashboard | Streamlit, Plotly |
| Model hosting | HuggingFace Hub |
| Deployment | Streamlit Community Cloud |

---

## Run locally

```bash
# Clone the repo
git clone https://github.com/JaySPX05/news-classifier.git
cd news-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo DATABASE_URL=your_supabase_url > .env

# Run the scraper
cd scrapers
python scraper.py

# Launch the dashboard
cd ..
streamlit run app.py
```

---

## Project structure
news-classifier/
app.py ← Streamlit dashboard
requirements.txt
scrapers/
scraper.py ← RSS scraper
db.py ← database helpers
label.py ← apply bias labels
sentiment.py ← VADER scoring
export.py ← export to CSV
migrate.py ← migrate to Supabase
models/
baseline.py ← TF-IDF + Logistic Regression
distilbert_classifier.py ← DistilBERT fine-tuning
compare.py ← model comparison
labels.py ← source → label mapping
data/
train.csv
val.csv
test.csv