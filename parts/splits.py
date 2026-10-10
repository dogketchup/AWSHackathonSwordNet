"""Train / dev / held-out assignment and nested data levels (scheduler plan, Part 1).

Pure logic: no storage, no files. Callers pass in the annotated images as dicts with at least
{"id", "filename", "origin"}.

Splits
    Every image belongs to a *group*: all frames of one video share a group, anything else is its
    own group. A hash of the group decides the split, so an image never changes split as the
    dataset grows, and near-identical frames of one video never sit on both sides.
        val  (VAL_FRACTION)  the held-out set; only the evaluator ever sees it
        dev  (DEV_FRACTION of the rest)  the trainer validates on it every epoch
        train  everything else

Levels
    A level is a snapshot of every annotated train/dev image at one moment. Annotation only adds
    images, so each level holds everything the previous one had (checked, and reported if not).
    The held-out set is frozen when the first level is made; held-out images annotated later are
    kept out of training but not added to the test, so every run is scored on the same images.
"""
import hashlib
import time
from dataclasses import asdict, dataclass, field

VAL_FRACTION = 0.20
DEV_FRACTION = 0.10


def group_key(img):
    origin = img.get("origin") or ""
    if origin.startswith("video:"):
        return origin.split("#", 1)[0]
    return img["filename"]


def _frac(key, salt=""):
    return int(hashlib.sha1((salt + key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(img, val_fraction=VAL_FRACTION, dev_fraction=DEV_FRACTION):
    key = group_key(img)
    if _frac(key) < val_fraction:
        return "val"
    return "dev" if _frac(key, "dev:") < dev_fraction else "train"


def assign(images, val_fraction=VAL_FRACTION, dev_fraction=DEV_FRACTION):
    """{"train": [ids], "dev": [ids], "val": [ids]} for the given images."""
    out = {"train": [], "dev": [], "val": []}
    for img in images:
        out[split_of(img, val_fraction, dev_fraction)].append(img["id"])
    for ids in out.values():
        ids.sort(key=str)
    return out


@dataclass
class Level:
    level_id: int
    created_at: float
    train: list
    dev: list
    missing_from_previous: list = field(default_factory=list)  # ids in the last level but not here

    @property
    def n_train(self):
        return len(self.train)


@dataclass
class LevelBook:
    """All levels made so far, plus the frozen held-out set."""
    levels: list = field(default_factory=list)
    heldout: list = field(default_factory=list)   # frozen at level 0
    min_new: int = 50                              # new train images needed for a new level
    min_heldout: int = 30                          # held-out images needed before freezing (team sets)

    def latest(self):
        return self.levels[-1] if self.levels else None

    def get(self, level_id):
        return next((l for l in self.levels if l.level_id == level_id), None)

    def maybe_new_level(self, annotated_images, now=None,
                        val_fraction=VAL_FRACTION, dev_fraction=DEV_FRACTION):
        """Return (new Level or None, reason). Follows the pseudocode's step 1."""
        s = assign(annotated_images, val_fraction, dev_fraction)
        if not self.levels and len(s["val"]) < self.min_heldout:
            return None, f"waiting: {len(s['val'])} held-out images, need {self.min_heldout} before freezing the test set"
        if not s["train"]:
            return None, "waiting: no training images yet"
        last = self.latest()
        if last is not None and len(s["train"]) - last.n_train < self.min_new:
            return None, f"waiting: {len(s['train']) - last.n_train} new training images since level {last.level_id}, need {self.min_new}"
        missing = sorted(set(last.train + last.dev) - set(s["train"] + s["dev"]), key=str) if last else []
        level = Level(level_id=len(self.levels), created_at=now if now is not None else time.time(),
                      train=s["train"], dev=s["dev"], missing_from_previous=missing)
        if not self.levels:
            self.heldout = s["val"]
        self.levels.append(level)
        reason = f"level {level.level_id}: {level.n_train} train / {len(level.dev)} dev"
        total = len(s["train"]) + len(s["dev"]) + len(s["val"])
        share = len(s["val"]) / total if total else 0
        if not 0.10 <= share <= 0.35:
            # Splits are per group (whole videos), so with few videos the held-out share can land far
            # from 20% (scaffold review M4). Reported, not corrected: moving groups would break stability.
            reason += f"; WARNING: held-out share is {share:.0%} of annotated images, target {val_fraction:.0%}"
        if missing:
            reason += f"; NOT NESTED: {len(missing)} images from level {last.level_id} are gone (deleted or sent back to labeled)"
        return level, reason

    def unused_heldout(self, annotated_images, val_fraction=VAL_FRACTION, dev_fraction=DEV_FRACTION):
        """Held-out-split images annotated after the freeze: never trained on, never scored."""
        frozen = set(self.heldout)
        return [i for i in assign(annotated_images, val_fraction, dev_fraction)["val"] if i not in frozen]

    def to_json(self):
        return {"levels": [asdict(l) for l in self.levels], "heldout": self.heldout,
                "min_new": self.min_new, "min_heldout": self.min_heldout}

    @classmethod
    def from_json(cls, d):
        return cls(levels=[Level(**l) for l in d.get("levels", [])], heldout=d.get("heldout", []),
                   min_new=d.get("min_new", 50), min_heldout=d.get("min_heldout", 30))
