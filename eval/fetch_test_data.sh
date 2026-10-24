#!/usr/bin/env bash
# Downloads the public plank test video used in docs/eval (MIT licence,
# github.com/NgoQuocBao1010/Exercise-Correction) into clips/.
set -euo pipefail
mkdir -p clips
curl -sSfL -o clips/plank.gif https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/images/plank.gif
cp eval/plank_gif_labels.csv clips/labels.csv
echo "Saved clips/plank.gif"
