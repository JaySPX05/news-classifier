import requests
from bs4 import BeautifulSoup

url = "https://www.bbc.com/news"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers, timeout=10)

print(f"Status code: {response.status_code}")
print(f"Page size: {len(response.text)} chars")

soup = BeautifulSoup(response.text, "html.parser")
headlines = soup.find_all("h3")

print(f"\nFound {len(headlines)} h3 elements. First 5:")
for h in headlines[:5]:
    text = h.get_text(strip=True)
    if text:
        print(" -", text)

for tag in ["h1", "h2", "h3", "h4", "a"]:
    found = soup.find_all(tag)
    texts = [el.get_text(strip=True) for el in found if len(el.get_text(strip=True)) > 30]
    print(f"\n{tag}: {len(texts)} results")
    for t in texts[:3]:
        print(f"  - {t}")

soup = BeautifulSoup(response.text, "html.parser")
headlines = soup.find_all("h2")

print(f"\nFound {len(headlines)} h2 elements. First 5:")
for h in headlines[:5]:
    text = h.get_text(strip=True)
    if text:
        print(" -", text)

{
    "name": "BBC",
    "url": "https://www.bbc.com/news",
    "tag": "h2",
    "class": None,
},