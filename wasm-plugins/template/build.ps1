# Build script for Extism WASM plugin
# Usage: .\build.ps1

$ErrorActionPreference = "Stop"

Write-Host "Building WASM plugin..." -ForegroundColor Cyan

# Ensure wasm32-unknown-unknown target is installed
Write-Host "Checking for wasm32-unknown-unknown target..." -ForegroundColor Yellow
$targets = rustup target list --installed
if ($targets -notcontains "wasm32-unknown-unknown") {
    Write-Host "Installing wasm32-unknown-unknown target..." -ForegroundColor Yellow
    rustup target add wasm32-unknown-unknown
}

# Build the plugin
Write-Host "Compiling to WebAssembly..." -ForegroundColor Yellow
cargo build --release --target wasm32-unknown-unknown

$wasmFile = "target/wasm32-unknown-unknown/release/plugin_template.wasm"
$pluginDir = "../../tauri-app/src-tauri/plugins/template"
$outputWasm = "$pluginDir/plugin.wasm"
$manifestFile = "$pluginDir/plugin.json"

# Check if build succeeded
if (-not (Test-Path $wasmFile)) {
    Write-Host "Build failed: WASM file not found" -ForegroundColor Red
    exit 1
}

# Create plugin directory
Write-Host "Creating plugin directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $pluginDir | Out-Null

# Copy WASM file
Write-Host "Copying WASM file..." -ForegroundColor Yellow
Copy-Item $wasmFile $outputWasm -Force

# Get WASM file size
$wasmSize = (Get-Item $wasmFile).Length
Write-Host "WASM file size: $([math]::Round($wasmSize/1KB, 2)) KB" -ForegroundColor Green

# Generate plugin manifest
Write-Host "Generating plugin manifest..." -ForegroundColor Yellow
$manifest = @{
    name = "template"
    version = "0.1.0"
    description = "Template plugin demonstrating Extism WASM capabilities"
    author = "Your Name"
    plugin_type = "utility"
    wasm_module = "plugin.wasm"
    wasm_config = @{
        allowed_hosts = @()
        allowed_paths = @()
        config = @{}
        memory_max_pages = 5
    }
    capabilities = @("json_processing", "host_functions")
    entry_points = @(
        @{
            name = "greet"
            function = "greet"
            description = "Generate a greeting message"
            input_format = "json"
            output_format = "json"
        },
        @{
            name = "repeat"
            function = "repeat"
            description = "Repeat a message N times"
            input_format = "json"
            output_format = "json"
        },
        @{
            name = "validate"
            function = "validate"
            description = "Validate input parameters"
            input_format = "json"
            output_format = "json"
        },
        @{
            name = "get_info"
            function = "get_info"
            description = "Get plugin metadata"
            input_format = "json"
            output_format = "json"
        }
    )
    dependencies = @{}
}

$manifest | ConvertTo-Json -Depth 10 | Out-File -FilePath $manifestFile -Encoding UTF8

Write-Host "Plugin built successfully!" -ForegroundColor Green
Write-Host "Location: $pluginDir" -ForegroundColor Cyan
Write-Host "  - plugin.wasm" -ForegroundColor Gray
Write-Host "  - plugin.json" -ForegroundColor Gray
Write-Host ""
Write-Host "To test the plugin:" -ForegroundColor Yellow
Write-Host "  1. Run the Tauri app: cd ../../tauri-app && pnpm dev" -ForegroundColor Gray
Write-Host "  2. Discover plugins in the UI" -ForegroundColor Gray
Write-Host "  3. Execute 'greet' function with: {""message"": ""World""}" -ForegroundColor Gray
