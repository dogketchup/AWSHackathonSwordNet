# Checker

## What it grades
Each training run is scored on a held-out validation split (20%, shuffled) that the run never trains on:
- **mAP50**: does the model find the sword (box overlaps the true box by at least 50%)?
- **mAP50-95**: is the oriented box tight and angled along the blade (overlap averaged from 50% to 95%)? This is the main score.
- **Overfitting**: train loss still falling while validation loss rises over the last epochs means the run is memorizing. The run's `best.pt` is taken from its best validation epoch, never the overfit tail, and runs stop after 10 epochs without improvement.
- **Video check**: on a sparring clip, how many frames get a sword box, and how many boxes are actually rotated (more than 10°) rather than upright.

## Why
Thin, diagonal blades are what the detector has to learn, so box tightness and angle (mAP50-95, rotated boxes) matter more than just finding something. A small validation set swings a lot between epochs, so differences of a point or two are treated as noise.

## The model's analysis
The workshop's default local model, local language model on our Trainium chip, read each run's metrics and wrote the analysis below. Copied unchanged from each run's `report.md` in the team model registry on the cluster:

### run-1
## Agent analysis ( on our Trainium chip)

- **Run comparison** – This run’s best mAP50‑95 (0.0844) is **worse** than earlier runs 4 (0.1039) and 5 (0.1005) and only marginally better than run 3 (0.0851). The 0.019‑point drop from run 5 is far larger than the ±5‑point noise expected from a 2‑image val set, so the run is not an improvement.  
- **Fit status** – Automatic findings flag **under‑fitting** (train and val loss still falling, 1.5 % per epoch). The curves are not converged; more epochs are needed before we can judge over‑/under‑fitting.  
- **Data quantity** – Only 19 sword objects in 13 training images (≈1.5 obj/img) and 3 objects in 2 val images. The dataset is far too small; adding 10–20 more labeled images (or more objects per image) will give the model a better chance to learn.  
- **Image size vs thin objects** – Blades are thin; at 640 px the sword occupies only a few dozen pixels. Increasing imgsz to 1280 or 1600 would give the model more resolution to learn the blade geometry.  
- **Label quality** – 22 manual labels for 22 objects suggests each image has a single sword. Verify that OBBs are tight and consistent; any sloppy boxes will hurt mAP.  
- **Val‑set leakage** – With only 2 val images, any single mislabeled or duplicate image can swing mAP by >5 %. Expand the val set to 5–10 images and ensure no overlap with train.  
- **Class balance** – Only the “sword” class is present. If sticks are also to be detected, add stick annotations; otherwise the model will learn a single‑class detector and may over‑fit to sword shape.  
- **Speed vs accuracy** – 19.5 fps on CPU (51 ms/img) is acceptable for offline inference but too slow for real‑time deployment. If speed is critical, consider a smaller backbone (yolo11n‑obb) or quantization after the model converges.  
- **Next investment** – Prioritize **more epochs** (e.g., 60–80) first, then **increase dataset size** and **imgsz**. If after more epochs the loss plateaus, shift focus to data augmentation and higher resolution.  
- **Action items for labelers** –  
  1. Add 10–20 new images with swords (and sticks if needed).  
  2. Double‑check OBB tightness and consistency.  
### run-2
## Agent analysis ( on our Trainium chip)

- **Performance jump** – best mAP 50‑95 of 0.2965 is ~0.19 higher than the best earlier run (0.1039), far above the ~5‑point noise expected from a 2‑image val set.  
- **Overfitting confirmed** – automatic findings flag overfitting: train loss decreases 0.078 / epoch while val loss rises 0.040 / epoch; val loss is lowest at epoch 1.  
- **Training data is tiny** – only 19 sword instances in 13 images; adding 20–30 more labeled swords would likely give a larger, more stable signal than extending epochs.  
- **Val set is too small** – 2 images can swing mAP by >10 points; expand to 10–20 images and keep a separate test set to avoid leakage.  
- **Image size vs thin objects** – swords are thin; 640 px may under‑sample blade edges. Try 1280 px (or 1024) to give more pixels per blade and improve localization.  
- **Augmentation** – add stronger geometric and photometric augmentations (random rotations, scaling, contrast jitter) to reduce overfitting and better generalise to unseen poses.  
- **Model‑size trade‑off** – 9.8 fps on CPU (101 ms/img) is acceptable for offline use; for real‑time deployment consider a smaller model (yolo11s‑obb) or GPU acceleration.  
- **Label quality check** – review the 22 manually‑labeled boxes for consistency (e.g., tightness, orientation) and ensure no duplicate or missing labels that could inflate training loss.  
- **Next run plan** – reduce epochs to ~15–20, increase batch size if GPU memory allows, add 50–100 new sword images, enlarge imgsz to 1280, and double the val set.  
- **Convergence status** – curves are inconclusive beyond the overfitting flag; no evidence of under‑fitting or convergence, so focus on data expansion and augmentation.

### run-3
## Agent analysis ( on our Trainium chip)

- **Performance vs. earlier runs** – Best mAP50‑95 of 0.0851 is **15 points lower** than the best 0.1005 (run 5) and 0.1039 (run 4).  The 15‑point drop is well above the ~5‑point noise expected from a 2‑image val set, so this run is objectively worse.  
- **Underfitting** – Both train (4.90) and val (12.62) losses are still falling (train‑loss slope –0.20, val‑loss slope –0.17) and the gap ratio (2.58) shows the val loss is still higher than the train loss.  The automatic finding flags underfitting.  
- **Regression warning** – Final mAP50‑95 (0.0709) is >10 % below the best epoch (0.0851).  Use `best.pt` for inference, not the last checkpoint.  
- **Dataset size** – Only 19 sword objects in 13 training images and 2 validation images.  The tiny val set means a single image can swing mAP by several points; a larger val set (≥10 images) is needed for stable metrics.  
- **Thin‑object resolution** – Blades are very narrow; a 640‑pixel image may not give enough pixels per blade.  Increasing `imgsz` to 1280 (or higher) would give more detail and likely improve detection of thin objects.  
- **Next investment – more data** – With only 19 objects, the model cannot learn robust features.  Prioritize adding more labeled swords (and sticks) and expanding the val set before extending training epochs.  
- **Next investment – more epochs** – If data expansion is delayed, extend training to 50–60 epochs (patience 20) to allow the loss curves to converge, as the current 30‑epoch run is still in the underfitting regime.  
- **Deployment considerations** – Current speed is 7.7 fps on CPU.  For real‑time use, switch to a GPU or use a lighter model (e.g., yolo11s‑obb) once the dataset is larger, to balance accuracy and inference speed.

### run-4
## Agent analysis ( on our Trainium chip)

- **Comparison to earlier yolo11s‑obb runs**  
  - Best mAP 50‑95 = **0.1039** (run 4) vs. 0.1005 (run 5), 0.0851 (run 3), 0.0844 (run 1).  
  - The 0.0034‑point gain is **< 5 % of the 2‑image val set**, so it is likely within val‑set noise rather than a real improvement.  
  - The only run that outperformed it (0.2965) used a larger model (yolo11m‑obb), so the improvement is not due to the model itself.

- **Training dynamics**  
  - Automatic findings flag **UNDERFITTING**: both train and val loss are still falling (train loss slope ≈ ‑0.064, val loss slope ≈ ‑0.200).  
  - Final train loss 4.88 vs. val loss 8.46 indicates the model has not yet saturated.

- **What to tweak next**  
  - **More epochs**: the loss curves are still descending; extending beyond 30 epochs should help.  
  - **More data**: only 19 sword objects in 13 images; adding ~10–20 more labeled swords would give the network more patterns to learn.  
### run-5
## Agent analysis ( on our Trainium chip)

- **Run comparison**  
  - Best mAP 50‑95: 0.1005 (run 5) vs 0.1039 (run 4), 0.0851 (run 3), 0.2965 (run 2), 0.0844 (run 1).  
  - Difference to run 4 is only 0.0034 (≈ 3.4 points), well below the ~5‑point noise margin for a 2‑image val set.  
  - Therefore, run 5 is **not statistically better** than run 4; the performance is essentially the same within val‑set noise.

- **Fit status**  
  - Automatic findings flag **UNDERFITTING**: both train and val loss are still falling (train loss slope ≈ –0.056, val loss slope ≈ –0.308).  
  - Curves are not converged; more training is needed.

- **Next‑step actions**  
  - **More epochs**: extend training beyond 30 epochs (patience 20 already set; consider 50–60 epochs).  
  - **Increase data**: add more sword images/annotations; 19 objects is very low.  
