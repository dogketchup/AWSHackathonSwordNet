# overview
need to build a fifty-fifty db on tritium hardware via a kubernetes bridge
for interacting with the DB the hardware a local webUI should ingest photos and videos, have user create Bounding Box labeling sets around swords and sticksd
webui should have upload tab, and annotate tab where as a distributed system, operators or webscraping agents can upload images, and labelers can annotate and mark ingest images as processed
webui should also have a webscraper integration and an autolabeler(grounding dino, ownlv2) integration
after labeling fifty-fifty should move files from ingest to labeled dir
20% of training should be allocated to validation
need a queuing system where as a user pulls an image to annotate from ingest

for kubernetes bridging, secrets need to be in a .env file, and there is **NO** push secrets tokens, etc into this repo

SAM should be run on ingested files to create a mask and opencv should be used to slim the OBB box down and fine tune accuracy for labeling\

On tritanium hardware local agent should kick off rolling trainings sessions on multiple yolo models and instance,
while the human labelers provide images and annotations.
each new run should handle grabbing all the ingested images
local tritanium Agent should write up a report on ("run count",yolo model(interested in small, medium, and large) , and how many training epochs, over fitting, and other ML metrics me as a human missed in thinking of)

should also have an easy way(.sh script and p1s script) to grab output weights, and load them in to a local deployment.

For human operators, I would like TUTORIAL.md files for each component so we can understand the architecture 



