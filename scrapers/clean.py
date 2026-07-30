import sqlite3
import re

DB_PATH = "D:/news-classifier/scrapers/data/headlines.db"

def clean_headline(text):
    text = text.strip()
    text = re.sub(r's+', ' ', text)        # collapse multiple spaces
    text = re.sub(r'^LIVEs*', '', text)    # remove LIVE prefix
    text = re.sub(r'd+ (hrs?|mins?) ago.*$', '', text)  # remove timestamps
    return text.strip()

def is_valid(text):
    if len(text) < 25:       return False   # too short to be a headline
    if len(text) > 300:      return False   # too long, probably a paragraph
    if text.isupper():       return False   # all caps = nav element
    if "cookie" in text.lower(): return False
    if "subscribe" in text.lower(): return False
    if "sign in" in text.lower():  return False
    return True

def clean_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    rows = conn.execute("SELECT id, headline FROM headlines").fetchall()
    print(f"Total rows before cleaning: {len(rows)}")

    deleted = 0
    updated = 0
    for row_id, headline in rows:
        cleaned = clean_headline(headline)
        if not is_valid(cleaned):
            cursor.execute("DELETE FROM headlines WHERE id = ?", (row_id,))
            deleted += 1
        elif cleaned != headline:
            cursor.execute(
                "UPDATE headlines SET headline = ? WHERE id = ?",
                (cleaned, row_id)
            )
            updated += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    print(f"Deleted:  {deleted} junk rows")
    print(f"Cleaned:  {updated} rows")
    print(f"Remaining: {total} headlines")
    conn.close()

if __name__ == "__main__":
    clean_db()