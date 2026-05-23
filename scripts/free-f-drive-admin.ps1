# free-f-drive-admin.ps1 — reclaim space on F:\ and harden WSL against
# the disk-full-induced VM crashes that have been recurring.
#
# REQUIRED: run this from an ELEVATED PowerShell window
# ("Run as administrator"). diskpart and `wsl --unregister` won't work
# without admin.
#
# What this does (all destructive ops are explicit + reported):
#   1. Confirms Docker Desktop is fully quit (you must quit it via the
#      tray icon yourself — its processes hold the VHDX open).
#   2. wsl --shutdown (clean stop of every WSL2 distro).
#   3. UNREGISTERS rancher-desktop and rancher-desktop-data
#      (per your decision: not actively used).
#   4. COMPACTS F:\WSL\Docker\DockerDesktopWSL\disk\docker_data.vhdx
#      in place via diskpart — preserves all Docker data, just
#      reclaims sparse-allocated empty space. This is the big win.
#   5. Tries to set the Ubuntu distro to sparse mode so its VHDX
#      auto-shrinks going forward (needs WSL 2.0+).
#   6. Reports F: free space before/after.
#
# Safe to re-run. If Docker Desktop is still running it bails clearly.

$ErrorActionPreference = 'Stop'

function Require-Admin {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Host "ERROR: this script must be run from an elevated PowerShell." -ForegroundColor Red
        Write-Host "       Right-click PowerShell -> 'Run as administrator', then re-run."
        exit 1
    }
}

function Get-FFreeGB {
    [math]::Round((Get-PSDrive F).Free / 1GB, 1)
}

Require-Admin

Write-Host "=== F:\ free space (BEFORE) ===" -ForegroundColor Cyan
$beforeGB = Get-FFreeGB
"  Free: $beforeGB GB"

# -- 1. Docker Desktop must be quit (we can't safely kill it; would corrupt VHDX) --
$dockerProcs = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(Docker Desktop|com\.docker)' }
if ($dockerProcs) {
    Write-Host "`nDocker Desktop is still running:" -ForegroundColor Yellow
    $dockerProcs | Select-Object Name, Id | Format-Table -AutoSize
    Write-Host "Please QUIT Docker Desktop (tray icon -> Quit Docker Desktop)," -ForegroundColor Yellow
    Write-Host "wait ~10 seconds for it to fully exit, then re-run this script."
    exit 2
}

# -- 2. wsl --shutdown (idempotent) --
Write-Host "`n=== wsl --shutdown ===" -ForegroundColor Cyan
wsl --shutdown
Start-Sleep -Seconds 3

# -- 3. Unregister Rancher Desktop distros (you said: not used) --
Write-Host "`n=== unregistering Rancher Desktop distros ===" -ForegroundColor Cyan
foreach ($d in @('rancher-desktop','rancher-desktop-data')) {
    $present = (wsl -l -q) -match "^$d$"
    if ($present) {
        Write-Host "  unregistering: $d"
        wsl --unregister $d
    } else {
        Write-Host "  (not registered: $d)"
    }
}

# -- 4. Compact Docker VHDX via diskpart --
$vhdx = 'F:\WSL\Docker\DockerDesktopWSL\disk\docker_data.vhdx'
if (-not (Test-Path $vhdx)) {
    Write-Host "`nNOT FOUND: $vhdx — skipping compaction." -ForegroundColor Yellow
} else {
    $vhdxGBBefore = [math]::Round((Get-Item $vhdx).Length/1GB, 1)
    Write-Host "`n=== compacting Docker VHDX (was $vhdxGBBefore GB) ===" -ForegroundColor Cyan

    $script = @"
select vdisk file="$vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
    $tmp = New-TemporaryFile
    $script | Set-Content -Path $tmp -Encoding ASCII
    try {
        diskpart /s $tmp
    } finally {
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
    $vhdxGBAfter = [math]::Round((Get-Item $vhdx).Length/1GB, 1)
    $reclaimed = [math]::Round($vhdxGBBefore - $vhdxGBAfter, 1)
    Write-Host "  VHDX size: $vhdxGBBefore GB -> $vhdxGBAfter GB (reclaimed $reclaimed GB)" -ForegroundColor Green
}

# -- 5. Set Ubuntu to sparse so its VHDX auto-shrinks going forward --
Write-Host "`n=== set Ubuntu distro sparse (auto-shrink future writes) ===" -ForegroundColor Cyan
try {
    wsl --manage Ubuntu --set-sparse true 2>&1 | Write-Host
} catch {
    Write-Host "  (--set-sparse not supported on this WSL version; OK to skip)" -ForegroundColor DarkGray
}

# -- 6. Final report --
Start-Sleep -Seconds 2
$afterGB = Get-FFreeGB
$totalReclaimed = [math]::Round($afterGB - $beforeGB, 1)
Write-Host "`n=== F:\ free space (AFTER) ===" -ForegroundColor Cyan
"  Free: $afterGB GB"
"  Net reclaimed this run: $totalReclaimed GB"

Write-Host "`nDONE." -ForegroundColor Green
Write-Host "If you re-launch Docker Desktop and start using it again, its VHDX will"
Write-Host "regrow up to whatever it actually needs (compact only removes EMPTY space)."
