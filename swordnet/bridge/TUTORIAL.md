# Bridge: driving the pods from the laptop

Our cluster role allows exactly two things: `get/list pods` and `exec` into them. No port-forward,
no Deployments/Jobs/Services. So the bridge ([`cli.py`](cli.py), [`kube.py`](kube.py)) does
everything with `kubectl exec`:

| Need | How |
|---|---|
| run a command | `kubectl exec seat-N -c app -- bash -c '...'` |
| copy files in | tar on the laptop → 1 MB pieces → `cat > piece` in the pod → reassemble → untar |
| copy files out | `tar c` + `split` in the pod → pieces over stdout → reassemble → untar on the laptop |
| long jobs | `setsid nohup ... &` in the pod, so they survive the exec session (and your Wi-Fi) |
| progress | poll a `status` file; tail the job's log into the central log |

Why pieces: measured on this cluster, one exec stream moves ~1 MB/s and gets cut off at random,
sometimes with exit code 0 and just fewer bytes. So every piece is checked against its own SHA-256
and retried, 6 pieces move at once, and the reassembled file is checked again. A 20 MB `s` weight
file takes ~40 s; an `l` run folder ~2 min.

Process checks use `kube.running()`, which brackets the first character of the pattern
(`[s]wordnet...`). Without that, `pgrep -f` inside `bash -c` matches its own shell and every job
looks alive forever.

Seat roles are in [`seats.py`](seats.py) (override with `SWORDNET_SEATS` JSON).

## Team registry ([`registry.py`](registry.py))

Teammates can't reach each other's laptops, but all of them can exec into the team seats, so
shared results live on seat-145 under `/workspace/swordnet-registry/`: every collected run's
`best.pt`, metrics, report and curves, an `index.json` listing them, a FiftyOne export of the
labeled data, and the raw videos. Runs are published automatically when collected; `publish`
(re)sends runs plus the dataset. Readers use `models`, `download` and `pull-dataset`; none of
those need the publisher's FiftyOne database.

Note that the registry is a shared copy, not a live database: labels made on another laptop
don't flow back. One laptop should be the labeling and training host.
Credentials come from `.env`, never from the repo.

```bash
B=".venv/bin/python -m swordnet.bridge.cli"
$B setup | status | push-code | autolabel | refine | cycle | loop | collect | report N | promote | pull-weights
$B publish | models | download [--run N] | pull-dataset        # the team registry, below
```
After changing code under `swordnet/`, run `push-code` so the pods get it.

Training keeps going on the pods if the bridge stops or your laptop drops off the network. When you
are back, `collect` pulls back every run the DB still shows as running and writes its report.
Piece transfers retry with backoff for about a minute, which covers a Wi-Fi or DNS blip.
