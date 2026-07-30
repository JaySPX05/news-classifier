# scrapers/test_rss.py
import requests
from bs4 import BeautifulSoup

feeds = [
    ("BBC", "http://feeds.bbci.co.uk/news/rss.xml"),
    ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    ("TheGuardian", "https://www.theguardian.com/world/rss"),
    ("FoxNews", "https://moxie.foxnews.com/google-publisher/latest.xml"),
]

for name, url in feeds:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    soup = BeautifulSoup(resp.content, "xml")
    titles = soup.find_all("title")
    print(f"\n{name} — {len(titles)} items")
    for t in titles[1:4]:
        print(f"  - {t.get_text(strip=True)}")