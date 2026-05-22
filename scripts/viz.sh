#!/usr/bin/env bash
# viz.sh — start the QFT-PCN visualizer backend and frontend together.
#
# Usage (from anywhere):
#     ./scripts/viz.sh
#
# Starts:
#   - the FastAPI backend via uvicorn on http://localhost:8000
#   - the React + Vite dev server via `pnpm dev` on http://localhost:5173
#
# Both run as background jobs; press Ctrl+C to stop both.
# Requires:  uv sync --extra viz   and   pnpm install  (in viz/web).

set -euo pipefail

# Resolve the repo root from this script's location so it works from any cwd.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
web_dir="$repo_root/src/qft_pcn/viz/web"

echo "Starting QFT-PCN visualizer..."
echo "  backend : http://localhost:8000"
echo "  frontend: http://localhost:5173"

# Backend: uvicorn FastAPI app, launched in the uv-managed environment.
( cd "$repo_root" && uv run uvicorn src.qft_pcn.viz.server:app --reload --port 8000 ) &
backend=$!

# Frontend: pnpm dev server.
( cd "$web_dir" && pnpm dev ) &
frontend=$!

# On exit (incl. Ctrl+C), tear both down.
cleanup() {
    kill "$backend" "$frontend" 2>/dev/null || true
    wait "$backend" "$frontend" 2>/dev/null || true
    echo "Visualizer stopped."
}
trap cleanup EXIT INT TERM

echo "Both processes running. Press Ctrl+C to stop."

# Wait until either process exits, then cleanup() tears the other one down.
wait -n "$backend" "$frontend"
