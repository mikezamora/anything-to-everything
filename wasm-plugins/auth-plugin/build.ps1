#!/usr/bin/env pwsh

Write-Host "Building authentication plugin..." -ForegroundColor Cyan

# Ensure wasm32-unknown-unknown target is installed
rustup target add wasm32-unknown-unknown

# Build the plugin
cargo build --target wasm32-unknown-unknown --release

if ($LASTEXITCODE -eq 0) {
    $wasmPath = "target/wasm32-unknown-unknown/release/auth_plugin.wasm"
    $size = (Get-Item $wasmPath).Length
    $sizeKB = [math]::Round($size / 1KB, 2)
    
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Write-Host "   Output: $wasmPath" -ForegroundColor Gray
    Write-Host "   Size: $sizeKB KB" -ForegroundColor Gray
    
    # Create plugin manifest
    $manifest = @{
        name = "auth-plugin"
        version = "0.1.0"
        description = "Authentication plugin with database host functions"
        author = "Tauri App"
        plugin_type = "service"
        wasm_module = "auth_plugin.wasm"
        wasm_config = @{
            allowed_hosts = @()
            allowed_paths = @{}
            config = @{}
            memory_max_pages = $null
        }
        capabilities = @()
        entry_points = @(
            @{
                name = "signup"
                function = "signup"
                description = "Create a new user account"
                input_format = "json"
                output_format = "json"
            },
            @{
                name = "login"
                function = "login"
                description = "Authenticate user and create session"
                input_format = "json"
                output_format = "json"
            },
            @{
                name = "verify_session"
                function = "verify_session"
                description = "Check if session is valid"
                input_format = "json"
                output_format = "json"
            },
            @{
                name = "logout"
                function = "logout"
                description = "End user session"
                input_format = "json"
                output_format = "json"
            },
            @{
                name = "get_current_user"
                function = "get_current_user"
                description = "Get current user information"
                input_format = "json"
                output_format = "json"
            },
            @{
                name = "verify_email"
                function = "verify_email"
                description = "Verify user email address"
                input_format = "json"
                output_format = "json"
            },
            @{
                name = "request_password_reset"
                function = "request_password_reset"
                description = "Request password reset"
                input_format = "json"
                output_format = "json"
            },
            @{
                name = "reset_password"
                function = "reset_password"
                description = "Reset user password"
                input_format = "json"
                output_format = "json"
            }
        )
        dependencies = @{}
    }
    
    $manifestJson = $manifest | ConvertTo-Json -Depth 10
    $manifestPath = "plugin.json"
    $manifestJson | Set-Content $manifestPath -Encoding UTF8
    
    Write-Host "   Manifest: $manifestPath" -ForegroundColor Gray
    
    # Copy to Tauri plugins directory
    $tauri_plugins_dir = "..\..\tauri-app\src-tauri\plugins"
    if (Test-Path $tauri_plugins_dir) {
        Copy-Item $wasmPath "$tauri_plugins_dir\auth-plugin.wasm" -Force
        Copy-Item $manifestPath "$tauri_plugins_dir\auth-plugin.json" -Force
        Write-Host "✅ Copied to Tauri plugins directory" -ForegroundColor Green
    }
    
    # Copy to AppData plugins directory
    $appdata_plugins_dir = "$env:APPDATA\anything-to-everything\plugins\auth-plugin"
    New-Item -ItemType Directory -Path $appdata_plugins_dir -Force | Out-Null
    Copy-Item $wasmPath "$appdata_plugins_dir\auth_plugin.wasm" -Force
    Copy-Item $manifestPath "$appdata_plugins_dir\plugin.json" -Force
    Write-Host "✅ Copied to AppData plugins directory" -ForegroundColor Green
    
} else {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}
