# scrapers/inspect_bytes.py
import sqlite3

conn = sqlite3.connect("D:/news-classifier/scrapers/data/headlines.db")
rows = conn.execute("SELECT headline FROM headlines WHERE source = 'BBC' LIMIT 5").fetchall()

for row in rows:
    h = row[0]
    print("TEXT:", h)
    print("BYTES:", h.encode("utf-8"))
    print()

conn.close()