import feedparser
import json
import requests
from datetime import datetime

# Free RSS feeds - no API key needed
RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_articles():
    all_articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        print(f"Fetching from {source_name}...")

        try:
            response = requests.get(feed_url, headers=HEADERS, timeout=10)
            print(f"  Status code: {response.status_code}")
            feed = feedparser.parse(response.content)
        except Exception as e:
            print(f"  Error fetching {source_name}: {e}")
            continue
        
        for entry in feed.entries:
            article = {
                "source": source_name,
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "fetched_at": datetime.now().isoformat(),
            }
            all_articles.append(article)

        print(f"  Got {len(feed.entries)} articles from {source_name}")

    return all_articles

if __name__ == "__main__":
    articles = fetch_articles()
    print(f"\nTotal articles fetched: {len(articles)}")

    with open("articles.json", "w") as f:
        json.dump(articles, f, indent=2)

    print("Saved to articles.json")