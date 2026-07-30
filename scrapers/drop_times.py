# scrapers/drop_times.py
import sqlite3

conn = sqlite3.connect("D:/news-classifier/scrapers/data/headlines.db")
conn.execute("DELETE FROM headlines WHERE source = 'TheTimes'")
conn.commit()
count = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
print("Done. Remaining rows: " + str(count))
conn.close()