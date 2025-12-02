# WASM Plugin Template

This is a template for creating Extism-compatible WebAssembly plugins that can be loaded into the Tauri application framework.

## Features

- **Extism PDK Integration**: Uses extism-pdk 1.0 for seamless plugin development
- **JSON Serialization**: Built-in support for JSON input/output via serde
- **Host Function Calls**: Demonstrates calling back to host functions
- **Error Handling**: Proper error handling with custom error codes
- **Optimized Build**: Configured for minimal WASM binary size
- **Auto-manifest Generation**: Build script generates plugin.json automatically

## Structure

```
template/
├── Cargo.toml           # Rust package configuration
├── src/
│   └── lib.rs          # Plugin implementation
├── build.ps1           # Build script for Windows
└── README.md           # This file
```

## Getting Started

### Prerequisites

- Rust toolchain (1.91.1 or later)
- wasm32-unknown-unknown target
- PowerShell (for build script)

### Building

```powershell
# From this directory
.\build.ps1
```

The build script will:
1. Install wasm32-unknown-unknown target if needed
2. Compile the plugin to WASM
3. Copy files to Tauri plugin directory
4. Generate plugin.json manifest
5. Display build results

### Testing

1. Start the Tauri app:
   ```powershell
   cd ../../tauri-app
   pnpm dev
   ```

2. In the UI:
   - Click "Discover Plugins"
   - Select "template" plugin
   - Choose "greet" function
   - Input: `{"message": "World"}`
   - Click "Execute Plugin"

Expected output:
```json
{
  "result": "Hello, World!",
  "timestamp": 0
}
```

## Plugin Functions

### greet

Generate a greeting message.

**Input:**
```json
{
  "message": "World",
  "count": 1
}
```

**Output:**
```json
{
  "result": "Hello, World!",
  "timestamp": 0
}
```

### repeat

Repeat a message N times.

**Input:**
```json
{
  "message": "Hello",
  "count": 3
}
```

**Output:**
```json
{
  "result": "Hello Hello Hello",
  "timestamp": 0
}
```

### validate

Validate input parameters.

**Input:**
```json
{
  "message": "Test",
  "count": 5
}
```

**Output:**
```json
{
  "result": "Valid input: Test",
  "timestamp": 0
}
```

**Errors:**
- Code 1: Message cannot be empty
- Code 2: Count must be between 1 and 100

### get_info

Get plugin metadata (version, functions, etc.).

**Input:** `{}` (any JSON)

**Output:**
```json
{
  "name": "Plugin Template",
  "version": "0.1.0",
  "functions": [...]
}
```

## Host Functions

The template demonstrates calling host functions:

```rust
#[host_fn]
extern "ExtismHost" {
    fn get_current_time() -> u64;
    fn log_message(message: String);
}
```

These are optional - the plugin works without them but can use them when available.

## Customization

### 1. Update Cargo.toml

```toml
[package]
name = "your-plugin-name"
version = "0.1.0"
description = "Your plugin description"
```

### 2. Implement Plugin Functions

```rust
#[plugin_fn]
pub fn your_function(Json(input): Json<YourInput>) -> FnResult<Json<YourOutput>> {
    // Your implementation
    Ok(Json(YourOutput { /* ... */ }))
}
```

### 3. Update build.ps1

Update the manifest generation section:
```powershell
$manifest = @{
    name = "your-plugin-name"
    version = "0.1.0"
    entry_points = @(
        @{
            name = "your_function"
            function = "your_function"
            description = "Description"
        }
    )
}
```

## Best Practices

1. **Keep it small**: WASM binaries should be as small as possible
   - Use `opt-level = "z"`
   - Enable `lto = true`
   - Add `strip = true`

2. **Error handling**: Always return proper errors
   ```rust
   return Err(WithReturnCode::new(
       Error::msg("Detailed error message"),
       error_code,
   ));
   ```

3. **JSON types**: Use strongly-typed structs with serde
   ```rust
   #[derive(Serialize, Deserialize)]
   pub struct Input {
       pub field: String,
   }
   ```

4. **Documentation**: Document all public functions
   ```rust
   /// Brief description
   /// 
   /// Detailed explanation
   #[plugin_fn]
   pub fn my_function(...) { }
   ```

## File Sizes

Expected WASM file sizes (release build):
- Template plugin: ~100-200 KB
- With dependencies: May increase based on crates used

To minimize size:
- Avoid large dependencies
- Use `default-features = false` where possible
- Profile with `cargo bloat --release --target wasm32-unknown-unknown`

## Troubleshooting

### Build fails with "target not installed"
```powershell
rustup target add wasm32-unknown-unknown
```

### Plugin not discovered
- Ensure plugin directory exists: `tauri-app/src-tauri/plugins/template/`
- Check plugin.json is valid JSON
- Verify wasm_module path is correct

### Function not found
- Check function name matches between:
  - `#[plugin_fn]` attribute
  - Entry point in plugin.json
  - Function call in UI

### Host function errors
Host functions are optional. The plugin handles their absence gracefully:
```rust
unsafe {
    get_current_time()
        .map(|t| t.as_u64())
        .unwrap_or(0)  // Fallback value
}
```

## Next Steps

1. **Create custom plugins**: Copy this template and modify for your needs
2. **Add host functions**: Implement host functions in Tauri backend
3. **Test thoroughly**: Test with various inputs and edge cases
4. **Deploy**: Copy to production plugin directory

## References

- [Extism Documentation](https://extism.org/docs)
- [Extism PDK Rust](https://github.com/extism/rust-pdk)
- [WebAssembly Spec](https://webassembly.github.io/spec/)
