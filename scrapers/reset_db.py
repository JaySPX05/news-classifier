# scrapers/reset_db.py
import sqlite3

conn = sqlite3.connect("D:/news-classifier/scrapers/data/headlines.db")
conn.execute("DELETE FROM headlines")
conn.commit()
count = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
print("Database cleared. Rows remaining: " + str(count))
conn.close()