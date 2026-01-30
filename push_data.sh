#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ubuntu/coinbase-ohlcv-data"
SRC_DATA_DIR="/home/ubuntu/coinbase-ohlcv/data"
DST_DATA_DIR="${REPO_DIR}/data"

cd "${REPO_DIR}"

mkdir -p "${DST_DATA_DIR}"

# Copy latest CSVs into the data repo
cp -f "${SRC_DATA_DIR}"/*.csv "${DST_DATA_DIR}/"

# Commit only if there are changes
git add "${DST_DATA_DIR}"/*.csv || true

if git diff --cached --quiet; then
  echo "No data change, skip git commit/push"
  exit 0
fi

git commit -m "Daily OHLCV update $(date -I)"
git push
echo "Pushed updated data to GitHub"
