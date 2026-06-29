#!/usr/bin/env bash
# Download GloVe 6B vectors and keep the 300d file the ventral route aligns to.
# Result: data/glove.6B.300d.txt  (auto-detected via config.DataConfig.glove_path)
#
#   bash data/get_glove.sh
#
# ~822 MB download; the extracted 300d file is ~1 GB. Both the zip and the txt
# are git-ignored.
set -euo pipefail
cd "$(dirname "$0")"                      # -> data/

PRIMARY="https://nlp.stanford.edu/data/glove.6B.zip"
MIRROR="https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"

if [ -f glove.6B.300d.txt ]; then
  echo "glove.6B.300d.txt already present — nothing to do."
  exit 0
fi

fetch() {
  if command -v curl >/dev/null 2>&1; then curl -L --fail -o glove.6B.zip "$1";
  elif command -v wget >/dev/null 2>&1; then wget -O glove.6B.zip "$1";
  else echo "Need curl or wget installed." >&2; exit 1; fi
}

echo "Downloading GloVe 6B (~822 MB) ..."
fetch "$PRIMARY" || { echo "Primary failed, trying HuggingFace mirror ..."; fetch "$MIRROR"; }

echo "Extracting glove.6B.300d.txt ..."
unzip -o glove.6B.zip glove.6B.300d.txt
rm -f glove.6B.zip
echo "Done -> data/glove.6B.300d.txt"
