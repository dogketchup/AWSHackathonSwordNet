"""Web scraper: page URLs or direct image URLs -> INGEST_DIR.

Give it a page and it downloads every <img> on it (skipping icons); give it an image URL and it
downloads that. Duplicates are dropped by content hash. Only scrape sites whose terms allow it.

Remote scraping agents don't need this module: they POST images to the web UI's /api/upload.
"""
import argparse
import hashlib
import io
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx
from PIL import Image

from swordnet import config
from swordnet.storage import db

MIN_SIDE = 160  # smaller than this is an icon or thumbnail
HEADERS = {"User-Agent": "SwordNet-scraper/0.1 (research dataset)"}


class _ImgParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.srcs = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            a = dict(attrs)
            src = a.get("src") or a.get("data-src")
            if src and not src.startswith("data:"):
                self.srcs.append(src)


def _image_urls(client, url):
    r = client.get(url)
    r.raise_for_status()
    if r.headers.get("content-type", "").startswith("image/"):
        return [url], {url: r.content}
    parser = _ImgParser()
    parser.feed(r.text)
    return [urljoin(url, s) for s in dict.fromkeys(parser.srcs)], {}


def _known_hashes(con):
    return {r["filename"].split("_", 1)[0] for r in con.execute(
        "SELECT filename FROM images WHERE filename LIKE 'scr%'")}


def scrape(urls, dataset, limit=200):
    config.ensure_dirs()
    added = []
    with httpx.Client(headers=HEADERS, timeout=20, follow_redirects=True) as client, db.session() as con:
        seen = _known_hashes(con)
        for page in urls:
            try:
                img_urls, cache = _image_urls(client, page)
            except httpx.HTTPError as e:
                print(f"skip {page}: {e}")
                continue
            for u in img_urls:
                if len(added) >= limit:
                    return added
                try:
                    data = cache.get(u) or client.get(u).content
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    continue
                if min(im.size) < MIN_SIDE:
                    continue
                h = "scr" + hashlib.sha1(data).hexdigest()[:16]
                if h in seen:
                    continue
                seen.add(h)
                name = f"{h}_.jpg"
                im.save(config.INGEST_DIR / name, quality=92)
                added.append(db.add_image(con, dataset, name, f"scrape:{u}", *im.size))
    return added


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("urls", nargs="+")
    p.add_argument("--dataset", default="default")
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args()
    print(f"added {len(scrape(a.urls, a.dataset, a.limit))} images")


if __name__ == "__main__":
    main()
