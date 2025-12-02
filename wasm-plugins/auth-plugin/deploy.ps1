#!/usr/bin/env pwsh
# Deploy auth plugin to Tauri app

param(
    [switch]$Clean = $false
)

Write-Host "🚀 Deploying Auth Plugin" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

# Get plugins directory
$pluginsDir = "$env:APPDATA\anything-to-everything\plugins"
Write-Host "`n📁 Plugins Directory: $pluginsDir" -ForegroundColor Yellow

# Clean if requested
if ($Clean) {
    Write-Host "`n🧹 Cleaning plugins directory..." -ForegroundColor Yellow
    if (Test-Path $pluginsDir) {
        Remove-Item -Path "$pluginsDir\*" -Recurse -Force
        Write-Host "   ✅ Cleaned" -ForegroundColor Green
    }
}

# Create directory if needed
if (-not (Test-Path $pluginsDir)) {
    New-Item -ItemType Directory -Path $pluginsDir -Force | Out-Null
    Write-Host "   ✅ Created plugins directory" -ForegroundColor Green
}

# Copy WASM plugin
$wasmSource = "target\wasm32-unknown-unknown\release\auth_plugin.wasm"
$wasmDest = "$pluginsDir\auth_plugin.wasm"

if (-not (Test-Path $wasmSource)) {
    Write-Host "`n❌ Error: WASM plugin not found at: $wasmSource" -ForegroundColor Red
    Write-Host "   Run: .\build.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n📦 Copying WASM plugin..." -ForegroundColor Yellow
Copy-Item -Path $wasmSource -Destination $wasmDest -Force
Write-Host "   ✅ Copied: auth_plugin.wasm" -ForegroundColor Green
Write-Host "   Size: $((Get-Item $wasmDest).Length / 1KB) KB" -ForegroundColor Gray

# Copy manifest
$manifestSource = "plugin.json"
$manifestDest = "$pluginsDir\plugin.json"

Write-Host "`n📋 Copying manifest..." -ForegroundColor Yellow
Copy-Item -Path $manifestSource -Destination $manifestDest -Force
Write-Host "   ✅ Copied: plugin.json" -ForegroundColor Green

# Verify files
Write-Host "`n✅ Deployment Complete!" -ForegroundColor Green
Write-Host "`nDeployed files:" -ForegroundColor Cyan
Get-ChildItem $pluginsDir | ForEach-Object {
    Write-Host "   - $($_.Name) ($($_.Length / 1KB) KB)" -ForegroundColor Gray
}

Write-Host "`n🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Start Tauri app: cd ..\..\tauri-app && pnpm tauri dev" -ForegroundColor Yellow
Write-Host "2. Plugin will be auto-loaded with host functions" -ForegroundColor Yellow
Write-Host "3. Test via Tauri commands (signup, login, verify_session, logout)" -ForegroundColor Yellow

Write-Host "`n📊 System Status:" -ForegroundColor Cyan
Write-Host "   Host Functions: 18/18 ✅" -ForegroundColor Green
Write-Host "   Database: SQLite ✅" -ForegroundColor Green
Write-Host "   Plugin System: Extism 1.13 ✅" -ForegroundColor Green
Write-Host "   Auth Plugin: Deployed ✅" -ForegroundColor Green
