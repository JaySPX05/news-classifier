import sqlite3

conn = sqlite3.connect("D:/news-classifier/scrapers/data/headlines.db")

total = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
by_source = conn.execute(
    "SELECT source, COUNT(*) as count FROM headlines GROUP BY source ORDER BY count DESC"
).fetchall()
oldest = conn.execute("SELECT MIN(scraped_at) FROM headlines").fetchone()[0]
newest = conn.execute("SELECT MAX(scraped_at) FROM headlines").fetchone()[0]

print(f"Total headlines: {total}")
print(f"Date range: {oldest} -> {newest}")
print("\nBy source:")
for source, count in by_source:
    bar = "█" * (count // 5)
    print(f"  {source:<12} {count:>4}  {bar}")

conn.close()