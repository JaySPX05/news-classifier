# scrapers/fix_encoding.py
import sqlite3
import re

conn = sqlite3.connect("D:/news-classifier/scrapers/data/headlines.db")
rows = conn.execute("SELECT id, headline FROM headlines").fetchall()

fixed = 0
for row_id, headline in rows:
    cleaned = headline.replace("\xa0", " ")  # non-breaking space
    cleaned = cleaned.replace("\u200b", "")  # zero width space
    cleaned = re.sub(r' {2,}', ' ', cleaned) # collapse multiple spaces
    cleaned = cleaned.strip()
    if cleaned != headline:
        conn.execute("UPDATE headlines SET headline = ? WHERE id = ?", (cleaned, row_id))
        fixed += 1

conn.commit()
print(f"Fixed {fixed} headlines")

sample = conn.execute("SELECT headline FROM headlines LIMIT 5").fetchall()
print("\nSample headlines after fix:")
for row in sample:
    print(" - " + row[0])

conn.close()