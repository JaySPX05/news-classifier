import requests
from bs4 import BeautifulSoup
from db import setup_db, insert_headline, get_headline_count

HEADERS = {"User-Agent": "Mozilla/5.0"}

SOURCES = [
    {
        "name": "BBC",
        "url": "http://feeds.bbci.co.uk/news/rss.xml",
    },
    {
        "name": "NPR",
        "url": "https://feeds.npr.org/1001/rss.xml",
    },
    {
        "name": "TheGuardian",
        "url": "https://www.theguardian.com/world/rss",
    },
    {
        "name": "FoxNews",
        "url": "https://moxie.foxnews.com/google-publisher/latest.xml",
    },
    {
        "name": "AlJazeera",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
    },
]

def scrape_source(source):
    try:
        resp = requests.get(source["url"], headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {source['name']}: {e}")
        return 0

    soup = BeautifulSoup(resp.content, "xml")
    titles = soup.find_all("title")

    inserted = 0
    for t in titles[1:]:
        text = t.get_text(strip=True)
        if len(text) > 20:
            if insert_headline(text, source["name"]):
                inserted += 1

    return inserted

def run():
    setup_db()
    print(f"Starting scrape. Current total: {get_headline_count()} headlines\n")
    for source in SOURCES:
        print(f"Scraping {source['name']}...")
        count = scrape_source(source)
        print(f"  Added {count} new headlines")
    print(f"\nDone. Total: {get_headline_count()} headlines")

if __name__ == "__main__":
    run()