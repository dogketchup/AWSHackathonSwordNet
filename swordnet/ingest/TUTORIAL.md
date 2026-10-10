# Ingest: getting images in

All three paths end the same way: a JPEG in `data/ingest/` and a FiftyOne sample with
`status=ingest`, the upload-set name, and an `origin` saying where it came from.

| Path | How | Code |
|---|---|---|
| Upload | Web UI → Upload tab (drag and drop images or videos) | `webui/app.py` `/api/upload` |
| Video frames | Upload a video; one frame every *N* s (default 0.5) | [`frames.py`](frames.py) |
| Scraper | Web UI → *Scrape a web page*, or `python -m swordnet.ingest.scraper URL...` | [`scraper.py`](scraper.py) |
| Agents | Any program can `POST /api/upload` (multipart `files`, form `dataset`) | |

Example for a scraping agent:
```bash
curl -F files=@sword.jpg -F dataset=scraped -H "X-Labeler: scraper-bot" http://<laptop-ip>:8080/api/upload
```

## Why frames remember their video

Frame `origin` is `video:<name>#<frame>`. The train/val split groups by video, so frames of one
clip are never on both sides. Without that, val mAP looks great because val frames are near-copies
of training frames.

## The scraper

It downloads every `<img>` on a page (or a direct image URL), skips anything under 160 px (icons),
and drops exact duplicates by content hash. **Only scrape sites whose terms allow it.**

## Auto-labeler ([`autolabel.py`](autolabel.py))

Zero-shot detectors that need no training: **OWLv2** (default) and **Grounding DINO**, prompted with
several phrasings per class ("a sword", "a longsword", "a katana", … / "a stick", "a wooden staff", …).
Overlapping hits from different phrasings are merged with NMS.

It runs only for upload sets whose **Auto-label** toggle is on (Upload tab → Datasets table):
```bash
.venv/bin/python -m swordnet.bridge.cli autolabel                  # OWLv2
.venv/bin/python -m swordnet.bridge.cli autolabel --model groundingdino --threshold 0.3
```
It runs on seat-149, so your laptop does not need PyTorch. Proposals show up **dashed** in the
editor. A labeler fixes or deletes them; *Save + mark processed* turns the kept ones into real labels.
