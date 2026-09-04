#!/usr/bin/env bash
# Pull the five runtime weights into ../weights/. Weights are NOT in git.
# Option A (recommended): upload the 5 files to a GitHub Release, put URLs below.
# Option B: copy from the shared Drive folder manually.
set -e
mkdir -p "$(dirname "$0")/../weights"
cd "$(dirname "$0")/../weights"
echo "Fetch these 5 files from the team Drive (amsdds/) or the GitHub Release:"
echo "  layer1_multimodal_mobilenetv3.pt      (17.7 MB)"
echo "  layer2_rtdetr.pt                      (142 MB)"
echo "  layer2_rtdetr_goa_head.pt             (0.6 MB)"
echo "  layer2_rtdetr_goa_head_hampad.pt      (3.0 MB)"
echo "  ood_scores_mobilenet.npz              (13.2 MB)"
# Example once a Release exists:
# curl -LO https://github.com/<org>/<repo>/releases/download/v1/layer2_rtdetr.pt
