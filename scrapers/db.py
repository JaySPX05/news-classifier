import sqlite3
import os

DB_PATH = "data/headlines.db"

def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def setup_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS headlines (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            headline    TEXT NOT NULL,
            source      TEXT NOT NULL,
            url         TEXT,
            scraped_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            bias_label  TEXT,          -- filled later in Phase 2
            sentiment   REAL           -- filled later in Phase 4
        )
    """)
    # Prevent storing the same headline twice
    cursor.execute("""
                 CREATE UNIQUE INDEX IF NOT EXISTS idx_headline_source
        ON headlines(headline, source)
    """)
    conn.commit()
    conn.close()
    print(f"Database ready at {DB_PATH}")

def insert_headline(headline, source, url=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO headlines (headline, source, url) VALUES (?, ?, ?)",
            (headline, source, url)
        )
        conn.commit()
        return True   # inserted
    except sqlite3.IntegrityError:
        return False  # duplicate, skipped
    finally:
        conn.close()

def get_headline_count():
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    conn.close()
    return count

if __name__ == "__main__":
    setup_db()