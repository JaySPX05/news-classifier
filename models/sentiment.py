# models/sentiment.py
import sqlite3
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DB_PATH = "D:/news-classifier/scrapers/data/headlines.db"

def score_headlines():
    analyzer = SentimentIntensityAnalyzer()
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("SELECT id, headline FROM headlines").fetchall()
    print(f"Scoring {len(rows)} headlines...")

    for row_id, headline in rows:
        scores = analyzer.polarity_scores(headline)
        compound = scores["compound"]  # -1 (most negative) to +1 (most positive)
        conn.execute(
            "UPDATE headlines SET sentiment = ? WHERE id = ?",
            (compound, row_id)
        )

    conn.commit()

    # Show average sentiment per source
    print("\nAverage sentiment by source:")
    rows = conn.execute("""
        SELECT source, 
               ROUND(AVG(sentiment), 3) as avg_sentiment,
               COUNT(*) as count
        FROM headlines
        GROUP BY source
        ORDER BY avg_sentiment DESC
    """).fetchall()

    for source, avg, count in rows:
        bar = "+" * int(avg * 20) if avg > 0 else "-" * int(abs(avg) * 20)
        print(f"  {source:<14} {avg:>7}  {bar}  ({count} headlines)")

    conn.close()

def score_single(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

if __name__ == "__main__":
    score_headlines()