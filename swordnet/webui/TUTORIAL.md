# Web UI

```bash
.venv/bin/uvicorn swordnet.webui.app:app --host 0.0.0.0 --port 8080
```
Open `http://localhost:8080`; teammates on the same network use `http://<laptop-ip>:8080`.
The first time, it asks your **name**: the queue and the logs use it.

## Tabs

**Upload**: drag in images or videos (videos become one frame every *N* s), scrape a page, and turn
**Auto-label** on/off per upload set.

**Annotate**
- *Queue* mode hands you the next image nobody else is working on. *Skip* gives it back.
- Drag to draw a box. There is one class, `sword` (key `1`). If more classes are added in
  `swordnet/config.py`, number keys choose the class of the **next** box; to relabel an existing
  box, click it first, then press its number. `Q`/`E` rotate the selected box by
  5°, `Del` deletes it, `Esc` deselects, `S` saves, `Enter` = **Save + mark processed** (moves the
  file from `ingest/` to `labeled/` and loads the next one).
- **Labeling rules:** box *every* sword that is visible, even partly, and include the whole
  visible blade. A frame with no sword in it is still useful: mark it processed with no boxes.
- Dashed boxes are auto-label proposals; a dot on a corner means SAM tightened it.
- An image with no swords is still useful: mark it processed with no boxes (a negative example).
- *Browse labeled* mode lets you go back and fix finished images.

**Reports**: every training run with mAP, FPS and status (the deployed run is badged); click one for
its report and training curves. Below: the raw files in `data/metrics/`.

**Logs**: everything from every component and pod, newest at the bottom, filterable by source
(`webui`, `bridge`, `agent`, `seat-146`, …), level, run number and text. *Follow* refreshes every 3 s.

## API (for agents and scripts)

| Method | Path | |
|---|---|---|
| POST | `/api/upload` | multipart `files`, form `dataset`, `every` |
| POST | `/api/scrape` | `{"urls": [...], "dataset": "..."}` |
| POST | `/api/queue/next` | claim the next image (header `X-Labeler`) |
| GET/PUT | `/api/images/{id}/labels` | pixel-coordinate boxes `{cls, points: [[x,y]*4]}` |
| POST | `/api/images/{id}/processed` | finish an image |
| GET | `/api/runs`, `/api/runs/{n}/report`, `/api/metrics`, `/api/logs` | |
