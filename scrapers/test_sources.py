# scrapers/test_sources.py
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

sites = [
    ("TheGuardian", "https://www.theguardian.com/world"),
    ("FoxNews", "https://www.foxnews.com"),
]

for name, url in sites:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    print(f"\n{name} — status {resp.status_code}")
    for tag in ["h1", "h2", "h3"]:
        found = [el.get_text(strip=True) for el in soup.find_all(tag) if len(el.get_text(strip=True)) > 25]
        print(f"  {tag}: {len(found)} results")
        for t in found[:2]:
            print(f"    - {t[:80]}")