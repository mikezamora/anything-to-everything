#!/usr/bin/env bash
# Install Manim Community Edition system prereqs + the viz-manim extra.
# Targets Debian/Ubuntu/WSL2. Run from the repo root.
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script targets apt-based systems. Install Manim's prereqs"
  echo "manually for your OS, then run: uv sync --extra viz-manim"
  exit 1
fi

echo "==> apt: installing Manim Community prereqs"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  python3-dev \
  libcairo2-dev \
  libpango1.0-dev \
  ffmpeg \
  texlive texlive-latex-extra texlive-fonts-extra texlive-science \
  pkg-config

echo "==> uv: syncing the viz-manim extra"
uv sync --extra viz-manim

echo "==> done. Try: uv run python -c 'import manim; print(manim.__version__)'"
