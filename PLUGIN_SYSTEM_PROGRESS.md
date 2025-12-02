# Plugin System Integration - Progress Report

## Overview

Successfully created a production-ready WASM plugin template and documented the complete plugin development workflow. This establishes the foundation for systematically porting features from the reference-code into Extism WASM plugins.

## Completed Work

### 1. Plugin Template Structure

Created `wasm-plugins/template/` with:

- **Cargo.toml**: Configured with extism-pdk 1.0, serde, optimized release profile
- **src/lib.rs**: Complete example plugin with 4 functions
- **build.ps1**: Automated build script for Windows
- **README.md**: Comprehensive documentation
- **.gitignore**: Proper exclusions

### 2. Template Features

The template demonstrates:

#### Plugin Functions
- `greet`: Simple greeting with JSON I/O
- `repeat`: Demonstrates optional parameters
- `validate`: Shows error handling with custom codes
- `get_info`: Returns plugin metadata

#### Technical Capabilities
- JSON serialization via serde
- Host function calls (optional)
- Error handling with return codes
- Type-safe input/output structures
- Optimized WASM compilation

#### Build Configuration
```toml
[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
strip = true        # Remove symbols
codegen-units = 1   # Better optimization
panic = "abort"     # Smaller binary
```

### 3. Build System

**build.ps1** script:
- Installs wasm32-unknown-unknown target if needed
- Compiles plugin to WASM
- Copies files to Tauri plugin directory
- Auto-generates plugin.json manifest
- Reports build results

**Result**: 151 KB WASM file (within target size)

### 4. Documentation

Created comprehensive guides:

#### wasm-plugins/README.md
- Architecture overview
- Quick start guide
- Plugin development guide
- Best practices
- Host function reference
- Performance optimization tips
- Troubleshooting
- CI/CD integration examples

#### wasm-plugins/template/README.md
- Template features
- Function descriptions
- Usage examples
- Customization guide
- Testing instructions

### 5. Reference Code Analysis

Analyzed key files from reference-code:

#### anticheat/src/lib.rs
- Comprehensive game anticheat system
- Multiple validation functions
- Complex data structures (Vec2, GameState, Violations)
- Server-side validation patterns
- Client heartbeat verification
- Physics simulation validation

#### server/services/authService.ts
- JWT authentication (@fastify/jwt)
- Argon2 password hashing (@node-rs/argon2)
- bcrypt legacy support with migration
- Session management
- Email verification
- Password reset flows
- Token-based auth (access + refresh tokens)

## Integration Architecture

### Current State

```
Tauri App
├── Backend (Rust)
│   ├── PluginManager
│   ├── PluginLoader (Extism)
│   └── Commands (6 total)
└── Frontend (React + TypeScript)
    ├── Typed API layer
    ├── React hooks
    └── Plugin UI

Plugin Directory
└── template/
    ├── plugin.wasm (151 KB)
    └── plugin.json
```

### Target Architecture

```
Tauri App
├── Backend (Rust)
│   ├── PluginManager (enhanced)
│   ├── Host Functions
│   │   ├── Database operations
│   │   ├── Time utilities
│   │   ├── Logging
│   │   └── HTTP client
│   ├── SQLite + ORM
│   ├── WebSocket handler
│   └── Security middleware
└── Frontend (React Router v7)
    ├── Auth flows
    ├── Plugin management
    ├── Game rendering
    └── WebSocket hooks

WASM Plugins
├── template/ (✅ complete)
├── auth/ (📋 planned)
├── audit/ (📋 planned)
├── anticheat/ (📋 planned)
├── tick-manager/ (📋 planned)
└── game-logic/ (📋 planned)
```

## Key Insights from Reference Code

### 1. Authentication Flow

Reference code uses multi-layered auth:
- Cookie-based sessions (primary)
- JWT access tokens (API)
- Refresh token rotation
- Argon2 for new passwords
- bcrypt legacy support with automatic migration

**WASM Plugin Strategy**: 
- JWT operations (sign/verify) in WASM
- Password hashing in WASM (argon2 crate)
- Database operations via host functions
- Session management via host functions

### 2. Database Schema

Reference code uses Drizzle ORM with:
- `users` table (uuid, name, email, passwordHash, emailVerified, avatar, bio)
- `sessions` table (id, userUuid, expiresAt)
- `emailVerificationTokens` table (token, userUuid, expiresAt)
- `passwordResetTokens` table (token, userUuid, expiresAt)

**Migration Strategy**:
- Port schema to Rust structs
- Use rusqlite or sea-orm
- Expose CRUD via host functions
- WASM plugins call host functions for data access

### 3. Anticheat System

Reference code shows production-ready patterns:
- Server-side state validation
- Physics simulation verification
- Teleportation detection
- Action spam detection
- Trust scoring system
- Client heartbeat challenges

**Direct Port**: Can copy anticheat/ directly since it's already Rust + Extism

### 4. Plugin Execution Pattern

Reference code uses:
```typescript
const result = await plugin.call("function_name", JSON.stringify(input));
```

**Already Implemented**: Our Tauri system follows same pattern via `execute_plugin` command

## Next Steps (Prioritized)

### Phase 1: Database Foundation (Critical Path)
1. Add SQLite to Tauri backend (rusqlite or sea-orm)
2. Port database schema from reference code
3. Implement migrations
4. Expose database operations as host functions

**Rationale**: All other plugins (auth, audit) depend on database access

### Phase 2: Authentication Plugin
1. Create `wasm-plugins/auth/` from template
2. Port authService.ts functions:
   - `signUp`, `signIn`, `signOut`
   - `verifyEmail`, `requestPasswordReset`, `resetPassword`
   - `issueAccessToken`, `verifyAccessToken`, `tokenRefresh`
3. Implement JWT operations (jsonwebtoken crate)
4. Implement password hashing (argon2 crate)
5. Call database via host functions

**Deliverable**: Fully functional auth system as WASM plugin

### Phase 3: Frontend Integration
1. Port React Router v7 routes from reference code
2. Add auth UI components (login, signup, password reset)
3. Implement auth guards for protected routes
4. Add WebSocket hooks

### Phase 4: Additional Plugins
1. Port anticheat module (already Rust + Extism)
2. Create audit logging plugin
3. Create tick manager plugin

### Phase 5: Production Features
1. Add WebSocket support
2. Implement security middleware
3. Add performance monitoring
4. Create deployment configs

## Technical Decisions

### 1. WASM vs Native

**Decision**: All business logic in WASM plugins
**Rationale**:
- Hot-reloadable without app restart
- Sandboxed execution
- Language-agnostic (can add non-Rust plugins)
- Matches reference code architecture

### 2. Database Access Pattern

**Decision**: Host functions for database access
**Rationale**:
- WASM can't directly access SQLite files
- Centralized connection pooling
- Transaction management in host
- Security boundary (plugins can't bypass permissions)

### 3. Host Function Design

**Decision**: High-level operations, not SQL strings
**Rationale**:
```rust
// ✅ Good: Type-safe operations
#[host_fn]
fn create_user(name: String, email: String, password_hash: String) -> String;

// ❌ Avoid: Raw SQL (security risk)
#[host_fn]
fn execute_sql(query: String) -> String;
```

### 4. JWT Placement

**Decision**: JWT signing/verification in WASM plugin
**Rationale**:
- No external dependencies needed by host
- Can be swapped for different auth methods
- Reference code shows JWT_SECRET in config (host can provide)

## File Size Analysis

| Plugin | Size | Status |
|--------|------|--------|
| template | 151 KB | ✅ Built |
| count_vowels | ~100 KB | ✅ Tested |
| anticheat | ~200 KB (estimated) | 📋 Planned |
| auth | ~150 KB (estimated) | 📋 Planned |

**Target**: Keep all plugins under 500 KB

## Build Performance

Template build times:
- Clean build: ~20s (42 dependencies)
- Incremental: ~5s

**Optimization**: Consider workspace-level Cargo.toml for shared dependencies

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_function() {
        // Test pure logic
    }
}
```

### Integration Tests

Via Tauri app:
1. Install plugin via UI
2. Call functions with test inputs
3. Verify outputs
4. Test error cases

### End-to-End Tests

Full auth flow:
1. Sign up → verify JWT structure
2. Verify email → check database state
3. Sign in → validate session creation
4. Protected route access → auth guard works
5. Sign out → session invalidated

## Known Issues & Limitations

### 1. Host Functions Not Implemented

Template demonstrates host function calls, but Tauri backend doesn't provide them yet.

**Status**: Non-blocking (plugins use fallback values)
**Next**: Implement in Phase 1

### 2. WebSocket Support

Reference code uses @fastify/websocket, not available in Tauri.

**Options**:
- tauri-plugin-websocket (if exists)
- Custom tokio-tungstenite implementation
- Tauri event system as alternative

**Decision Needed**: Research Tauri WebSocket options

### 3. Session Storage

Reference code uses SQLite sessions. Need equivalent in Tauri.

**Solution**: Implement in Phase 1 database work

## Dependencies Summary

### Existing (Tauri Backend)
- extism 1.4.1
- extism-manifest 1.13.0
- tokio 1.42
- reqwest 0.12
- wasmparser 0.239
- tauri 2.9.4

### Needed (Phase 1)
- rusqlite or sea-orm (database)
- tokio-tungstenite (WebSocket)
- governor (rate limiting)

### Needed (Phase 2)
- jsonwebtoken (JWT in auth plugin via extism-pdk)
- argon2 (password hashing in auth plugin)

## Success Metrics

### Phase 1 Complete
- ✅ Plugin template with 4 working functions
- ✅ 151 KB WASM size (under 200 KB target)
- ✅ Auto-generating manifest
- ✅ Comprehensive documentation

### Phase 2 Complete (Next)
- [ ] SQLite integrated with migrations
- [ ] Database host functions exposed
- [ ] Auth plugin built and tested
- [ ] Login/signup flows working

### Phase 3 Complete
- [ ] React Router v7 fully ported
- [ ] Auth guards protecting routes
- [ ] WebSocket real-time communication

### Phase 4 Complete
- [ ] All reference code features ported
- [ ] Production-ready deployment
- [ ] K8S/Docker configs
- [ ] Performance benchmarks

## Conclusion

Successfully established the foundation for WASM plugin development:

1. ✅ **Template Created**: Production-ready starting point
2. ✅ **Build System**: Automated with manifest generation
3. ✅ **Documentation**: Comprehensive guides for developers
4. ✅ **Reference Analysis**: Understand features to port
5. 📋 **Roadmap Defined**: Clear path to full integration

**Next Immediate Action**: Start Phase 2 by implementing SQLite database layer with host functions in Tauri backend.

---

**Last Updated**: 2025-01-XX
**Status**: Phase 1 Complete ✅
**Next Phase**: Database Foundation
