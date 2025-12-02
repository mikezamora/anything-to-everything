# Phase 3: Host Functions for WASM Plugins

## Status: In Progress ⚠️

Host functions have been **partially implemented** but require completion due to Extism 1.13 API complexity.

## What Was Done

### ✅ Module Structure Created
- `src-tauri/src/host_functions/mod.rs` - Main module with `register_host_functions()` 
- `src-tauri/src/host_functions/database.rs` - 18 database host function stubs

### ✅ Integration Points Completed
- `PluginManager` updated with `database: Option<Arc<Database>>` field
- `PluginManager::new_with_database()` constructor added
- `PluginLoader::load_with_host_functions()` method added
- Host functions passed to plugins during load

### ⚠️ Implementation Challenge: Extism 1.13 Memory API

The Extism 1.13 API for host functions uses a complex memory model:
- `MemoryHandle` requires both offset AND length (not exposed to host functions)
- `CurrentPlugin::memory_get()` expects typed `FromBytes` trait implementations
- Simple byte array access is not straightforward

## Current Host Function Signatures

All 18 database operations have function signatures defined:

### User Operations (6)
1. `db_create_user` - Create new user
2. `db_get_user_by_email` - Find user by email
3. `db_get_user_by_uuid` - Find user by UUID
4. `db_update_user_password` - Update password hash
5. `db_update_user_email_verified` - Mark email verified
6. `db_update_user_profile` - Update profile fields

### Session Operations (5)
7. `db_create_session` - Create session with expiration
8. `db_get_session` - Get session by ID
9. `db_delete_session` - Delete specific session
10. `db_delete_user_sessions` - Delete all user sessions
11. `db_cleanup_expired_sessions` - Remove expired sessions

### Email Verification (3)
12. `db_create_email_verification_token` - Create token
13. `db_get_email_verification_token` - Get token
14. `db_delete_email_verification_token` - Delete token

### Password Reset (4)
15. `db_create_password_reset_token` - Create token
16. `db_get_password_reset_token` - Get token
17. `db_delete_password_reset_token` - Delete token
18. `db_delete_user_password_reset_tokens` - Delete all user tokens

## Request/Response Structures

All request and response types are defined:
- `CreateUserRequest`
- `UpdateUserProfileRequest` 
- `UpdatePasswordRequest`
- `UpdateEmailVerifiedRequest`
- `CreateSessionRequest`
- `CreateEmailVerificationTokenRequest`
- `CreatePasswordResetTokenRequest`
- `HostResponse<T>` - Generic success/error response

## What Needs to Be Fixed

### Option 1: Fix Extism 1.13 Implementation

The memory helpers need to correctly use the Extism 1.13 API:

```rust
// Current broken approach
fn read_json<T>(plugin: &mut CurrentPlugin, offset: i64) -> Result<T, String> {
    let handle = extism::MemoryHandle::new(offset as u64); // ❌ Private + needs length
    let bytes: Vec<u8> = plugin.memory_get(handle)?; // ❌ Wrong trait bounds
    serde_json::from_slice(&bytes)?
}
```

**Correct approach** (needs research):
- Use `plugin.input_get()` for reading host function inputs
- Use `plugin.memory_new()` correctly for allocating output
- May need to use Extism's built-in JSON conversion traits

### Option 2: Use Host Function with String I/O

Simpler approach using Extism's built-in string conversion:

```rust
pub fn create_user_host(state: Arc<HostFunctionState>) -> Function {
    Function::new(
        "db_create_user",
        [ValType::I64],
        [ValType::I64],
        UserData::new(Arc::new(Mutex::new(state))),
        |plugin, inputs, outputs, user_data| {
            // Use Extism's string helpers
            let input_str: String = plugin.input_get(/* ... */)?;
            let request: CreateUserRequest = serde_json::from_str(&input_str)?;
            
            // Process...
            
            let response_str = serde_json::to_string(&response)?;
            let handle = plugin.memory_new(response_str)?;
            outputs[0] = Val::I64(handle.offset() as i64);
            Ok(())
        }
    )
}
```

### Option 3: Downgrade to Extism 1.4

If Extism 1.13 API is too complex, downgrade to 1.4 which has simpler memory API:
- Update `Cargo.toml`: `extism = "1.4"`
- Use simpler `plugin.memory_get(offset)` that returns `&[u8]`
- Use simpler `plugin.memory_new(&[u8])` that returns offset directly

## Compilation Status

**Current:** ❌ Does not compile
- Error: `MemoryHandle` is private
- Error: `MemoryHandle::new()` requires 2 arguments (offset, length)
- Error: `memory_get()` trait bounds not satisfied

**To compile:** Comment out or remove `host_functions/database.rs` temporarily

## Next Steps

1. **Research Extism 1.13 host function examples** - Find working examples in Extism repo
2. **Test with simple host function** - Implement one function (e.g., `db_test_connection`) 
3. **Once pattern works, apply to all 18 functions** - Use macro to reduce boilerplate
4. **Alternative: Use Extism 1.4** if 1.13 API is too complex for our use case

## Testing Plan (Once Working)

1. **Create test WASM plugin** that calls host functions:
   ```rust
   // In plugin
   #[host_import("extism:host/user")]
   extern "ExtismHost" {
       fn db_create_user(ptr: u64) -> u64;
   }
   ```

2. **Test each operation**:
   - Create user → verify in DB
   - Get user by email → check returned data
   - Create session → verify expiration
   - Token operations → test lifecycle

3. **Integration test**: Full auth flow through WASM plugin

## Current Workaround

For now, plugins can be loaded **without** host functions:
- `PluginManager::new()` works (sets `database: None`)
- Plugins can export functions that the host calls
- Host-to-plugin direction works (what we have now)
- Plugin-to-host direction (host functions) needs completion

## Documentation

When complete, add to plugin development guide:
- How to import host functions in WASM plugins
- Request/response JSON schemas
- Error handling patterns
- Example authentication plugin using host functions
