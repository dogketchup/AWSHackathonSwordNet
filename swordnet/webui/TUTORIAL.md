# Web UI

```bash
.venv/bin/uvicorn swordnet.webui.app:app --host 0.0.0.0 --port 8080
```
Open `http://localhost:8080`. The first time, it asks your **name**: the queue and the logs use it.

## One UI for the whole team

Everyone should use the **same** web UI, the one on the host laptop, so there is one queue, one
set of labels and one training history. Teammates open `http://<host-laptop-ip>:8080`
(`hostname -I` shows it):

1. **Set a team password first.** Put `SWORDNET_UI_PASSWORD=<something>` in the host's `.env` and
   restart the server. Browsers then ask once (any username). Without it, anyone on the network
   can label, delete images or change settings. Scripts pass it with `curl -u team:<password>`.
2. **Same network.** Teammates must be on the host's Wi-Fi or hotspot. From elsewhere, put the
   host and the teammates on a private Tailscale network and use the host's Tailscale address.
   Don't expose the UI with a public tunnel.
3. **Firewall.** If the page times out for others: `sudo ufw allow 8080/tcp` on the host.

Pods can't serve the UI: the workshop role has no port-forward. Trained models are also shared
through the registry on seat-145, which works without the host laptop (`bridge models`).

## Tabs

**Upload**: drag in images or videos (videos become one frame every *N* s), scrape a page, and turn
**Auto-label** on/off per upload set.

**Annotate**
- *Queue* mode hands you the next image nobody else is working on. *Skip* gives it back.
- Drag to draw a box. `1` sword / `2` stick choose the class of the **next** box you draw. To
  relabel an existing box, click it first, then press `1`/`2`. `Q`/`E` rotate the selected box by
  5°, `Del` deletes it, `Esc` deselects, `S` saves, `Enter` = **Save + mark processed** (moves the
  file from `ingest/` to `labeled/` and loads the next one).
- **Labeling rules:** box *every* sword and stick that is visible, even partly; include the whole
  visible blade; sword = steel/metal blade, stick = wooden waster, bamboo shinai or plain stick.
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
