# Build script for audit plugin
Write-Host "Building audit plugin..." -ForegroundColor Green

# Build the plugin
cargo build --release --target wasm32-unknown-unknown

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Get the output file
$wasmFile = "target\wasm32-unknown-unknown\release\audit_plugin.wasm"

if (!(Test-Path $wasmFile)) {
    Write-Host "WASM file not found: $wasmFile" -ForegroundColor Red
    exit 1
}

# Display file size
$fileSize = (Get-Item $wasmFile).Length
$fileSizeKB = [math]::Round($fileSize / 1KB, 2)
Write-Host "Built: $wasmFile ($fileSizeKB KB)" -ForegroundColor Green

# Copy to plugins directory
$pluginsDir = "..\..\tauri-app\src-tauri\plugins"
if (!(Test-Path $pluginsDir)) {
    New-Item -ItemType Directory -Path $pluginsDir -Force | Out-Null
}

Copy-Item $wasmFile "$pluginsDir\audit-plugin.wasm" -Force
Write-Host "Copied to: $pluginsDir\audit-plugin.wasm" -ForegroundColor Green

# Copy manifest
Copy-Item "plugin.json" "$pluginsDir\audit-plugin.json" -Force
Write-Host "Copied manifest to: $pluginsDir\audit-plugin.json" -ForegroundColor Green

# Copy to AppData plugins directory
$appdata_plugins_dir = "$env:APPDATA\anything-to-everything\plugins\audit-plugin"
New-Item -ItemType Directory -Path $appdata_plugins_dir -Force | Out-Null
Copy-Item $wasmFile "$appdata_plugins_dir\audit_plugin.wasm" -Force
Copy-Item "plugin.json" "$appdata_plugins_dir\plugin.json" -Force
Write-Host "Copied to AppData: $appdata_plugins_dir" -ForegroundColor Green

Write-Host "`nAudit plugin build complete!" -ForegroundColor Green
