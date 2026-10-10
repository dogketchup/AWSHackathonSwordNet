"""Sword-clash demo: track swords in a video and play a sound whenever two sword boxes meet.

The model finds the swords in each frame; Ultralytics' tracker (ByteTrack, which runs a Kalman
filter per sword) keeps an ID on each one across frames. When two tracked swords' boxes go from
apart to overlapping, that's a clash: one sound, then a short cooldown for that pair.

    python scripts/clash_demo.py VIDEO --weights best.pt              # -> VIDEO_clash.mp4 with sound
    python scripts/clash_demo.py VIDEO --weights best.pt --live       # window + sound as it plays
    python scripts/clash_demo.py VIDEO --weights yoloe-11l-seg.pt     # no trained model? text prompts
    --sound my.wav   use your own sound instead of the generated clink.wav

Works with detect and OBB (rotated-box) models; OBB clashes use the rotated boxes' real overlap.
"""
import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

SR = 44100
PROMPTS = ["a sword", "a stick", "a fencing sword", "a long thin blade", "a metal rod", "a hockey stick", "a weapon", "a cane"]


def make_clink(path, seed=7):
    """A metallic 'shing': two blades ringing at slightly mismatched inharmonic partials, a hard
    strike transient on top, and a quick shimmer so it doesn't sound like a plain bell."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * 0.9)) / SR
    out = np.zeros_like(t)
    for base, gain in ((1840.0, 1.0), (1913.0, 0.8)):                 # blade A, blade B
        for ratio, amp, decay in ((1.0, 1.0, 7), (2.76, 0.55, 11), (5.40, 0.35, 18), (8.93, 0.2, 30)):
            f = base * ratio
            out += gain * amp * np.exp(-decay * t) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    out *= 1 + 0.25 * np.sin(2 * np.pi * 23 * t)                        # shimmer
    strike = rng.normal(0, 1, t.size) * np.exp(-t / 0.004)              # metal-on-metal tick
    out = out / np.abs(out).max() * 0.8 + strike * 0.35
    out = np.tanh(out * 1.3) * 0.9                                      # a little grit
    _write_wav(path, out)


def _write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767
        ch, sr = w.getnchannels(), w.getframerate()
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        data = np.interp(np.linspace(0, len(data), int(len(data) * SR / sr), endpoint=False), np.arange(len(data)), data)
    return data


def load_model(weights):
    if "yoloe" in Path(weights).name:
        from ultralytics import YOLOE
        m = YOLOE(weights)
        m.set_classes(PROMPTS, m.get_text_pe(PROMPTS))
        return m
    from ultralytics import YOLO
    return YOLO(weights)


def sword_shapes(r):
    """[(track_id or None, polygon Nx2 float32)] for this frame."""
    out = []
    if getattr(r, "obb", None) is not None and r.obb is not None and len(r.obb):
        ids = r.obb.id.int().tolist() if r.obb.id is not None else [None] * len(r.obb)
        for tid, poly in zip(ids, r.obb.xyxyxyxy.cpu().numpy()):
            out.append((tid, poly.astype(np.float32)))
    elif r.boxes is not None and len(r.boxes):
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for tid, (x1, y1, x2, y2) in zip(ids, r.boxes.xyxy.cpu().numpy()):
            out.append((tid, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)))
    return out


def overlap(a, b, same=0.8):
    """Two swords touch if their boxes overlap, unless one box lies almost entirely inside the other
    (that's two boxes on the same sword, not a clash)."""
    area, _ = cv2.intersectConvexConvex(a, b)
    if area <= 0:
        return False
    smaller = min(cv2.contourArea(a), cv2.contourArea(b))
    return smaller <= 0 or area / smaller < same


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--weights", default="yoloe-11l-seg.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--cooldown", type=float, default=0.4, help="seconds before the same pair can clash again")
    ap.add_argument("--sound", help="a .wav to use instead of the generated clink")
    ap.add_argument("--out", help="output video (default: VIDEO_clash.mp4)")
    ap.add_argument("--live", action="store_true", help="show a window and play sounds as it runs")
    a = ap.parse_args()

    video = Path(a.video)
    out = Path(a.out) if a.out else video.with_name(video.stem + "_clash.mp4")
    sound = Path(a.sound) if a.sound else out.with_name("clink.wav")
    if not a.sound:
        make_clink(sound)
    clink = _read_wav(sound)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    model = load_model(a.weights)

    tmp_video = Path(tempfile.mkstemp(suffix=".mp4")[1])
    vw = None if a.live else cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    audio = np.zeros(int((n / fps + 2) * SR), np.float32)
    touching, last_clash, clashes, flash = set(), {}, [], 0

    for i, r in enumerate(model.track(str(video), stream=True, persist=True, conf=a.conf,
                                      agnostic_nms=True, iou=0.5, tracker="bytetrack.yaml", verbose=False)):
        frame, t = r.orig_img.copy(), i / fps
        shapes = sword_shapes(r)
        now_touching = set()
        for x in range(len(shapes)):
            for y in range(x + 1, len(shapes)):
                if overlap(shapes[x][1], shapes[y][1]):
                    ia, ib = shapes[x][0], shapes[y][0]
                    pair = (min(ia, ib), max(ia, ib)) if None not in (ia, ib) else ("any",)
                    now_touching.add(pair)
                    if pair not in touching and t - last_clash.get(pair, -9) >= a.cooldown:
                        last_clash[pair] = t
                        clashes.append(t)
                        flash = int(fps * 0.25)
                        s0 = int(t * SR)
                        audio[s0:s0 + len(clink)] += clink[: len(audio) - s0]
                        if a.live:
                            subprocess.Popen(["afplay", str(sound)])
        touching = now_touching

        for tid, poly in shapes:
            col = (0, 0, 255) if any(tid in p for p in now_touching) else (60, 220, 60)
            cv2.polylines(frame, [poly.astype(np.int32)], True, col, 3)
            if tid is not None:
                cv2.putText(frame, f"sword {tid}", tuple(poly[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        if flash > 0:
            cv2.putText(frame, "CLASH!", (w // 2 - 120, 90), cv2.FONT_HERSHEY_DUPLEX, 2.4, (0, 0, 255), 5)
            flash -= 1
        cv2.putText(frame, f"clashes: {len(clashes)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        if a.live:
            cv2.imshow("SwordNet clash demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            vw.write(frame)

    if a.live:
        cv2.destroyAllWindows()
    else:
        vw.release()
        peak = np.abs(audio).max()
        wav = out.with_name(out.stem + "_audio.wav")
        _write_wav(wav, audio / peak * 0.9 if peak > 1 else audio)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", str(wav),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
        wav.unlink(missing_ok=True)
        tmp_video.unlink(missing_ok=True)
        print(f"{len(clashes)} clashes at {[round(c, 2) for c in clashes]} s -> {out}")


if __name__ == "__main__":
    main()
