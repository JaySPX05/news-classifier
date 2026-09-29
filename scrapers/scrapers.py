from email.mime import text
import requests
from bs4 import BeautifulSoup
from db import setup_db, insert_headline, get_headline_count, insert_to_supabase

HEADERS = {"User-Agent": "Mozilla/5.0"}

SOURCES = [
    { "name": "BBC",         "url": "http://feeds.bbci.co.uk/news/rss.xml" },
    { "name": "NPR",         "url": "https://feeds.npr.org/1001/rss.xml" },
    { "name": "TheGuardian", "url": "https://www.theguardian.com/world/rss" },
    { "name": "FoxNews",     "url": "https://moxie.foxnews.com/google-publisher/latest.xml" },
    { "name": "AlJazeera",   "url": "https://www.aljazeera.com/xml/rss/all.xml" },
    { "name": "TheHill",     "url": "https://thehill.com/rss/syndicator/19110" },
    { "name": "Vox",         "url": "https://www.vox.com/rss/index.xml" },
    { "name": "Breitbart",   "url": "https://feeds.feedburner.com/breitbart" },
    { "name": "Politico",    "url": "https://rss.politico.com/politics-news.xml" },
    { "name": "NYTimes",     "url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml" },
    { "name": "WSJ",         "url": "https://feeds.a.dj.com/rss/RSSWorldNews.xml" },
    { "name": "CNN",         "url": "http://rss.cnn.com/rss/edition.rss" },
    { "name": "TheHindu",      "url": "https://www.thehindu.com/news/feeder/default.rss" },
    { "name": "NDTV",          "url": "https://feeds.feedburner.com/ndtvnews-top-stories" },
    { "name": "IndiaToday",    "url": "https://www.indiatoday.in/rss/home" },
    { "name": "TimesOfIndia",  "url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms" },
    { "name": "HindustanTimes", "url": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml" },
    { "name": "TheWire",       "url": "https://thewire.in/feed" },
    { "name": "OpIndia",       "url": "https://www.opindia.com/feed/" },
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
                insert_to_supabase(text, source["name"])
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

import time

if __name__ == "__main__":
    while True:
        run()
        print("\nSleeping for 3 hours...")
        time.sleep(3 * 60 * 60)  # 3 hours in seconds