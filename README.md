# SwordNet: project overview

**Goal:** Build a sword and stick detection pipeline. It has a FiftyOne dataset and model training on AWS
Trainium hardware, reached from a local web UI through a Kubernetes bridge.

## 1. Data and storage

- One FiftyOne database holds all data, labels and run history.
- It runs on the Trainium machine and is reached through a Kubernetes bridge.
- Secrets (AWS keys, tokens) live only in a `.env` file. No secrets or tokens are ever committed or pushed to
  the repo.

## 2. Web UI (runs locally)

- **Upload tab:** operators, or web-scraping agents, upload photos and videos into an ingest folder.
- **Annotate tab:** labelers draw oriented bounding boxes around swords and sticks, then mark each image as done.
- **Built-in tools:** a web scraper, and an auto-labeler (Grounding DINO or OWLv2) that suggests boxes before a
  human reviews them.
- **Queue:** each labeler pulls the next image from ingest. No two labelers get the same image, so many people
  can work at once.
- **On save:** the file moves from ingest to labeled, and the label is stored in FiftyOne.

## 3. Label refinement

- SAM runs on each ingested image and makes a mask of the object.
- OpenCV fits a tight rotated box to that mask, so labels are more accurate with less manual adjustment.

## 4. Training on Trainium

- A local agent keeps running training rounds on several YOLO models (small, medium and large), with repeated
  runs of each, while labelers keep adding data.
- Each new round picks up all newly labeled images.
- 20% of the data is held out for validation.

## 5. Agents

- **Qwen3-8B** runs deployment and testing: it plans the runs, launches training and picks the best model.
- **gpt-oss-20b** writes the analysis report for each round.

## 6. Metrics and reports

- Metrics scripts live in a `metrics/` folder, and the web UI can read their output.
- Each report covers:
  - run count, model size (S/M/L) and number of epochs
  - overfitting
  - any other ML issues a human might miss
- Reports and logs can be viewed in the web UI.

## 7. Using the trained model

- `.sh` (Linux/macOS) and `.ps1` (Windows) scripts download the latest weights and load them into a local
  deployment.

## 8. Documentation

- Each component has a `TUTORIAL.md` that explains how it works and how it fits into the overall architecture,
  for human operators.
