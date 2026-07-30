import sqlite3
import sys
sys.path.append("../models")
from labels import SOURCE_LABELS

DB_PATH = "D:/news-classifier/scrapers/data/headlines.db"

def apply_labels():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total_updated = 0
    for source, label in SOURCE_LABELS.items():
        cursor.execute(
            "UPDATE headlines SET bias_label = ? WHERE source = ?",
            (label, source)
        )
        updated = cursor.rowcount
        total_updated += updated
        print(f"  {source:<14} -> {label:<8} ({updated} rows)")

    conn.commit()

    print(f"Total labeled: {total_updated}")
    print("Label distribution:")
    rows = conn.execute(
        "SELECT bias_label, COUNT(*) FROM headlines GROUP BY bias_label ORDER BY COUNT(*) DESC"
    ).fetchall()
    for label, count in rows:
        bar = "X" * (count // 5)
        print(f"  {label:<8} {count:>4}  {bar}")

    conn.close()

if __name__ == "__main__":
    apply_labels()