# WASM Plugins

This directory contains WebAssembly plugins for the Anything-to-Everything framework. All plugins are compiled to WASM and loaded dynamically via the Extism runtime.

## Architecture

```
wasm-plugins/
├── template/          # Plugin template for creating new plugins
├── auth/              # Authentication plugin (JWT, passwords, sessions)
├── audit/             # Audit logging plugin
└── anticheat/         # Game anticheat plugin (from reference-code)
```

Each plugin is a standalone Rust project that compiles to a `.wasm` file and includes a `plugin.json` manifest.

## Quick Start

### Creating a New Plugin

1. **Copy the template:**
   ```powershell
   cp -r template my-plugin
   cd my-plugin
   ```

2. **Update Cargo.toml:**
   ```toml
   [package]
   name = "my-plugin"
   description = "My custom plugin"
   ```

3. **Implement your functions:**
   ```rust
   #[plugin_fn]
   pub fn my_function(Json(input): Json<MyInput>) -> FnResult<Json<MyOutput>> {
       // Your logic here
       Ok(Json(MyOutput { /* ... */ }))
   }
   ```

4. **Build:**
   ```powershell
   .\build.ps1
   ```

5. **Test in Tauri app:**
   - Run: `cd ../tauri-app && pnpm dev`
   - Discover plugins in UI
   - Execute your function

### Building All Plugins

```powershell
# From this directory
Get-ChildItem -Directory | ForEach-Object {
    if (Test-Path "$($_.FullName)\build.ps1") {
        Push-Location $_.FullName
        .\build.ps1
        Pop-Location
    }
}
```

## Plugin Development Guide

### Project Structure

Each plugin follows this structure:

```
my-plugin/
├── Cargo.toml          # Rust configuration
├── src/
│   └── lib.rs          # Plugin implementation
├── build.ps1           # Build script
├── README.md           # Plugin documentation
└── .gitignore          # Git ignore file
```

### Required Dependencies

```toml
[dependencies]
extism-pdk = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[lib]
crate-type = ["cdylib"]
```

### Plugin Function Signature

All exported functions use the `#[plugin_fn]` attribute:

```rust
use extism_pdk::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Input {
    pub field: String,
}

#[derive(Serialize, Deserialize)]
pub struct Output {
    pub result: String,
}

#[plugin_fn]
pub fn my_function(Json(input): Json<Input>) -> FnResult<Json<Output>> {
    Ok(Json(Output {
        result: format!("Processed: {}", input.field),
    }))
}
```

### Host Functions

Plugins can call back to the host for shared functionality:

```rust
#[host_fn]
extern "ExtismHost" {
    fn get_current_time() -> u64;
    fn log_message(message: String);
    fn query_database(query: String) -> String;
}

#[plugin_fn]
pub fn with_host_call(Json(input): Json<Input>) -> FnResult<Json<Output>> {
    // Call host function
    let timestamp = unsafe {
        get_current_time().unwrap_or(0)
    };
    
    // Use the result
    Ok(Json(Output {
        result: format!("Timestamp: {}", timestamp),
    }))
}
```

### Error Handling

Return errors with custom codes:

```rust
#[plugin_fn]
pub fn validate(Json(input): Json<Input>) -> FnResult<Json<Output>> {
    if input.field.is_empty() {
        return Err(WithReturnCode::new(
            Error::msg("Field cannot be empty"),
            1, // Error code
        ));
    }
    
    Ok(Json(Output {
        result: "Valid".to_string(),
    }))
}
```

### Manifest Generation

Each plugin needs a `plugin.json` manifest:

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Plugin description",
  "plugin_type": "utility",
  "wasm_module": "plugin.wasm",
  "wasm_config": {
    "allowed_hosts": [],
    "allowed_paths": [],
    "memory_max_pages": 5
  },
  "entry_points": [
    {
      "name": "my_function",
      "function": "my_function",
      "description": "Function description",
      "input_format": "json",
      "output_format": "json"
    }
  ]
}
```

The build script (`build.ps1`) generates this automatically.

## Best Practices

### 1. Keep Plugins Small

- Minimize dependencies
- Use `opt-level = "z"` for size optimization
- Enable LTO and strip symbols
- Target < 500 KB for most plugins

### 2. Use Type Safety

```rust
// ✅ Good: Strongly typed
#[derive(Serialize, Deserialize)]
pub struct UserInput {
    pub username: String,
    pub email: String,
}

// ❌ Avoid: Generic JSON
pub fn my_function(Json(input): Json<serde_json::Value>) { }
```

### 3. Handle Errors Gracefully

```rust
// ✅ Good: Descriptive errors with codes
if user.is_none() {
    return Err(WithReturnCode::new(
        Error::msg("User not found"),
        404,
    ));
}

// ❌ Avoid: Generic errors
return Err(Error::msg("Error"));
```

### 4. Document Functions

```rust
/// Authenticate a user with email and password
/// 
/// # Arguments
/// * `email` - User's email address
/// * `password` - User's password (will be hashed)
/// 
/// # Returns
/// * `AuthResult` - Contains JWT token on success
/// 
/// # Errors
/// * Code 401: Invalid credentials
/// * Code 403: Account not verified
#[plugin_fn]
pub fn authenticate(Json(input): Json<AuthInput>) -> FnResult<Json<AuthResult>> {
    // Implementation
}
```

### 5. Test Thoroughly

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_input() {
        let input = MyInput {
            field: "test".to_string(),
        };
        assert_eq!(input.field, "test");
    }
    
    #[test]
    fn test_invalid_input() {
        let input = MyInput {
            field: "".to_string(),
        };
        assert!(input.field.is_empty());
    }
}
```

## Plugin Categories

### Utility Plugins

General-purpose functionality:
- String processing
- Data validation
- Format conversion

### Authentication Plugins

User management and security:
- JWT token generation
- Password hashing
- Session management
- OAuth integration

### Database Plugins

Data persistence:
- CRUD operations
- Query building
- Migration management

### Game Plugins

Game-specific logic:
- Physics simulation
- Collision detection
- Anticheat systems
- Tick management

### Integration Plugins

External service connections:
- HTTP clients
- Webhook handlers
- API wrappers

## Host Function Reference

### Available Host Functions

(To be implemented in Tauri backend)

```rust
// Time utilities
fn get_current_time() -> u64;
fn format_timestamp(ts: u64) -> String;

// Logging
fn log_message(message: String);
fn log_error(error: String);
fn log_warning(warning: String);

// Database
fn query_database(query: String) -> String;
fn execute_database(query: String) -> i64;

// Storage
fn read_file(path: String) -> Vec<u8>;
fn write_file(path: String, data: Vec<u8>) -> bool;

// HTTP
fn http_get(url: String) -> String;
fn http_post(url: String, body: String) -> String;

// Events
fn emit_event(event: String, data: String);
fn subscribe_event(event: String);
```

## Performance Tips

### 1. Optimize Cargo Configuration

```toml
[profile.release]
opt-level = "z"     # Optimize aggressively for size
lto = true          # Link-time optimization
strip = true        # Remove debug symbols
codegen-units = 1   # Better optimization
panic = "abort"     # Smaller binary
```

### 2. Use `wasm-opt`

```powershell
# Install
cargo install wasm-opt

# Optimize
wasm-opt -Oz -o plugin_optimized.wasm plugin.wasm
```

### 3. Profile Binary Size

```powershell
# Install
cargo install cargo-bloat

# Analyze
cargo bloat --release --target wasm32-unknown-unknown
```

### 4. Lazy Initialization

```rust
use std::sync::OnceLock;

static EXPENSIVE_RESOURCE: OnceLock<Resource> = OnceLock::new();

#[plugin_fn]
pub fn use_resource(Json(input): Json<Input>) -> FnResult<Json<Output>> {
    let resource = EXPENSIVE_RESOURCE.get_or_init(|| {
        // Expensive initialization only happens once
        Resource::new()
    });
    
    // Use resource
    Ok(Json(Output { /* ... */ }))
}
```

## Troubleshooting

### Build Errors

**Error: target not installed**
```powershell
rustup target add wasm32-unknown-unknown
```

**Error: linker error**
```powershell
# Update Rust
rustup update stable
```

### Runtime Errors

**Error: function not found**
- Check function name in `#[plugin_fn]`
- Verify entry_points in plugin.json
- Rebuild with `.\build.ps1`

**Error: invalid JSON**
- Validate input JSON structure
- Check serde Serialize/Deserialize derives
- Use `serde_json::to_string_pretty` for debugging

### Size Issues

**WASM file too large**
- Remove unused dependencies
- Check `cargo bloat` output
- Use `wasm-opt -Oz`
- Consider splitting into multiple plugins

## CI/CD Integration

### GitHub Actions

```yaml
name: Build WASM Plugins

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
      
      - name: Build all plugins
        run: |
          for dir in wasm-plugins/*/; do
            if [ -f "$dir/build.sh" ]; then
              cd "$dir"
              ./build.sh
              cd -
            fi
          done
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: wasm-plugins
          path: tauri-app/src-tauri/plugins/*/plugin.wasm
```

### Local Pre-commit Hook

```powershell
# .git/hooks/pre-commit
#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

Write-Host "Building WASM plugins..." -ForegroundColor Cyan

Get-ChildItem wasm-plugins -Directory | ForEach-Object {
    if (Test-Path "$($_.FullName)\build.ps1") {
        Push-Location $_.FullName
        Write-Host "Building $($_.Name)..." -ForegroundColor Yellow
        .\build.ps1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build failed for $($_.Name)" -ForegroundColor Red
            exit 1
        }
        Pop-Location
    }
}

Write-Host "All plugins built successfully" -ForegroundColor Green
```

## Resources

- [Extism Documentation](https://extism.org/docs)
- [Extism Rust PDK](https://github.com/extism/rust-pdk)
- [WebAssembly Spec](https://webassembly.github.io/spec/)
- [Rust WASM Book](https://rustwasm.github.io/docs/book/)
- [WASM Optimization Guide](https://rustwasm.github.io/book/reference/code-size.html)

## Contributing

When adding a new plugin:

1. Copy the template
2. Implement functionality
3. Add tests
4. Document in README.md
5. Update this file's architecture section
6. Submit PR with:
   - Plugin source code
   - Build script
   - Documentation
   - Example usage

## License

All plugins in this directory follow the same license as the main project.
