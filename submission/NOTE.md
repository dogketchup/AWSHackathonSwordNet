# One-page note

**Task.** Detect swords in sparring footage with tight, oriented boxes (YOLO11-OBB), trained on the AWS Trainium seat pods of the workshop cluster. The local language model on our Trainium chip reads every run's metrics and writes the analysis; a checker (`CHECKER.md`) grades each run on a held-out 20% split.

**What we ran (all on the Kubernetes cluster, see `attempt_log.csv`).**
1. **Round 1, human labels:** 5 runs, mAP50-95 0.084–0.296 (mean 0.134, spread 0.212). 13 training / 2 validation images, so scores swing widely; the analysis flagged under-fitting, too little data and too few validation images.
2. **Round 2, SAM-oriented labels:** 46 sparring clips sampled at 1 frame/s (1,305 frames). The round-1 model found swords in 362 frames; SAM + OpenCV turned each box into a rotated box along the blade (427 of 430 boxes). Shuffled 80/20 into 290 train (+1 motion-blurred copy each) / 72 validation. YOLO11s-OBB trained in parallel on four seats at four image sizes: 4 runs, mAP50-95 0.173–0.197 (mean 0.188, spread 0.024).
3. **Round 3, continued training** from each seat's best checkpoint: 3 runs, mAP50-95 0.122–0.183 (mean 0.157, spread 0.061).

**Result.** Best run: `seat-148/obb-416` at image size 416, mAP50 0.398, mAP50-95 0.197. On a held-out sparring video it found a sword in 156 of 194 frames, and 74% of its boxes were rotated along the blade (the round-1 model's boxes were all upright). A Kalman filter on the tracked boxes fills short gaps from blur, raising coverage further.

**Spread and caveats.** Differences between round-2 runs are within a few points on a 72-image validation set, so we treat them as close; image size 416 trained fastest and led on mAP50-95. Round-2 labels are machine-made (detector + SAM) and not human-checked, so scores measure agreement with those labels; misses are mostly dark blades on dark backgrounds, crossed swords, and a matte-floor gym underrepresented in the data. Next: more labeled frames from that gym, human-corrected boxes, and longer training.
