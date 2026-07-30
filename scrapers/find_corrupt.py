# scrapers/find_corrupt.py
import sqlite3

conn = sqlite3.connect("D:/news-classifier/scrapers/data/headlines.db")
rows = conn.execute("SELECT headline, source FROM headlines").fetchall()

corrupt = {}
for headline, source in rows:
    if "  " in headline:
        corrupt[source] = corrupt.get(source, 0) + 1

print("Corrupted headlines by source:")
for source, count in sorted(corrupt.items(), key=lambda x: -x[1]):
    print(f"  {source:<14} {count}")

conn.close()