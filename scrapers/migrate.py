# scrapers/migrate.py
import sqlite3
import os
from sqlalchemy import create_engine, text

DATABASE_URL = st.secrets["DATABASE_URL"]

SQLITE_PATH = "D:/news-classifier/scrapers/data/headlines.db"

def setup_table(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS headlines (
                id          SERIAL PRIMARY KEY,
                headline    TEXT NOT NULL,
                source      TEXT NOT NULL,
                url         TEXT,
                scraped_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                bias_label  TEXT,
                sentiment   REAL,
                UNIQUE(headline, source)
            )
        """))
        conn.commit()
    print("Table created successfully")

def migrate(engine):
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    rows = sqlite_conn.execute(
        "SELECT headline, source, url, scraped_at, bias_label, sentiment FROM headlines"
    ).fetchall()
    sqlite_conn.close()

    print(f"Migrating {len(rows)} headlines to Supabase...")

    inserted = 0
    skipped = 0
    with engine.connect() as conn:
        for row in rows:
            try:
                conn.execute(text("""
                    INSERT INTO headlines (headline, source, url, scraped_at, bias_label, sentiment)
                    VALUES (:headline, :source, :url, :scraped_at, :bias_label, :sentiment)
                    ON CONFLICT (headline, source) DO NOTHING
                """), {
                    "headline":   row[0],
                    "source":     row[1],
                    "url":        row[2],
                    "scraped_at": row[3],
                    "bias_label": row[4],
                    "sentiment":  row[5],
                })
                inserted += 1
            except Exception as e:
                skipped += 1
        conn.commit()

    print(f"Done. Inserted: {inserted}, Skipped: {skipped}")

if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    setup_table(engine)
    migrate(engine)