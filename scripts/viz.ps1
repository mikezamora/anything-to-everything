# viz.ps1 — start the QFT-PCN visualizer backend and frontend together.
#
# Usage (from the repo root):
#     ./scripts/viz.ps1
#
# Starts:
#   - the FastAPI backend via uvicorn on http://localhost:8000
#   - the React + Vite dev server via `pnpm dev` on http://localhost:5173
#
# Both run as background jobs; press Ctrl+C (or close the window) to stop.
# Requires:  uv sync --extra viz   and   pnpm install  (in viz/web).

$ErrorActionPreference = "Stop"

# Resolve the repo root from this script's location so it works from any cwd.
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$webDir = Join-Path $repoRoot "src/qft_pcn/viz/web"

Write-Host "Starting QFT-PCN visualizer..." -ForegroundColor Cyan
Write-Host "  backend : http://localhost:8000" -ForegroundColor DarkGray
Write-Host "  frontend: http://localhost:5173" -ForegroundColor DarkGray

# Backend: uvicorn FastAPI app, launched in the uv-managed environment.
$backend = Start-Process -PassThru -NoNewWindow -WorkingDirectory $repoRoot `
    -FilePath "uv" `
    -ArgumentList "run", "uvicorn", "src.qft_pcn.viz.server:app", "--reload", "--port", "8000"

# Frontend: pnpm dev server.
$frontend = Start-Process -PassThru -NoNewWindow -WorkingDirectory $webDir `
    -FilePath "pnpm" -ArgumentList "dev"

Write-Host "Both processes running. Press Ctrl+C to stop." -ForegroundColor Green

try {
    # Wait until either process exits, then tear the other one down.
    while (-not $backend.HasExited -and -not $frontend.HasExited) {
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($p in @($backend, $frontend)) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "Visualizer stopped." -ForegroundColor Yellow
}
