# Phase 2: Database Foundation - Implementation Guide

## Overview

Add SQLite database support to Tauri backend and expose database operations as host functions for WASM plugins.

## Goals

1. Integrate SQLite with Tauri backend
2. Port database schema from reference-code
3. Implement migrations
4. Expose host functions for WASM plugins
5. Test database operations from plugins

## Architecture

```
Tauri Backend
├── db/
│   ├── mod.rs          # Database module
│   ├── schema.rs       # Table definitions
│   ├── migrations.rs   # Migration system
│   └── operations.rs   # CRUD operations
└── host_functions/
    └── database.rs     # Host functions for plugins
```

## Database Schema (from reference-code)

### Users Table
```rust
pub struct User {
    pub id: i64,
    pub uuid: String,
    pub name: String,
    pub email: String,
    pub password_hash: String,
    pub email_verified: bool,
    pub avatar: Option<String>,
    pub bio: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}
```

### Sessions Table
```rust
pub struct Session {
    pub id: String,          // UUID
    pub user_uuid: String,
    pub created_at: i64,
    pub expires_at: i64,
}
```

### Email Verification Tokens
```rust
pub struct EmailVerificationToken {
    pub token: String,
    pub user_uuid: String,
    pub created_at: i64,
    pub expires_at: i64,
}
```

### Password Reset Tokens
```rust
pub struct PasswordResetToken {
    pub token: String,
    pub user_uuid: String,
    pub created_at: i64,
    pub expires_at: i64,
}
```

## Implementation Steps

### Step 1: Add Dependencies

Update `tauri-app/src-tauri/Cargo.toml`:

```toml
[dependencies]
# Existing dependencies...

# Database
rusqlite = { version = "0.32", features = ["bundled"] }
uuid = { version = "1.0", features = ["v4"] }
chrono = "0.4"
```

### Step 2: Create Database Module

**tauri-app/src-tauri/src/db/mod.rs:**

```rust
use rusqlite::{Connection, Result};
use std::path::PathBuf;

pub mod schema;
pub mod migrations;
pub mod operations;

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(db_path: PathBuf) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(Database { conn })
    }
    
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}
```

### Step 3: Implement Migrations

**tauri-app/src-tauri/src/db/migrations.rs:**

```rust
use rusqlite::{Connection, Result};

pub fn run_migrations(conn: &Connection) -> Result<()> {
    // Create version table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at INTEGER NOT NULL
        )",
        [],
    )?;
    
    let current_version = get_schema_version(conn)?;
    
    if current_version < 1 {
        migrate_v1(conn)?;
    }
    
    Ok(())
}

fn get_schema_version(conn: &Connection) -> Result<i32> {
    conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version",
        [],
        |row| row.get(0),
    )
    .unwrap_or(Ok(0))
}

fn migrate_v1(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            email_verified INTEGER NOT NULL DEFAULT 0,
            avatar TEXT,
            bio TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY (user_uuid) REFERENCES users(uuid)
        );
        
        CREATE TABLE email_verification_tokens (
            token TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY (user_uuid) REFERENCES users(uuid)
        );
        
        CREATE TABLE password_reset_tokens (
            token TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY (user_uuid) REFERENCES users(uuid)
        );
        
        INSERT INTO schema_version (version, applied_at) 
        VALUES (1, strftime('%s', 'now'));"
    )?;
    
    Ok(())
}
```

### Step 4: Database Operations

**tauri-app/src-tauri/src/db/operations.rs:**

```rust
use rusqlite::{Connection, Result, params};
use crate::db::schema::*;

// User operations
pub fn create_user(
    conn: &Connection,
    uuid: &str,
    name: &str,
    email: &str,
    password_hash: &str,
    created_at: i64,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO users (uuid, name, email, password_hash, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
        params![uuid, name, email, password_hash, created_at],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn get_user_by_email(conn: &Connection, email: &str) -> Result<Option<User>> {
    let mut stmt = conn.prepare(
        "SELECT id, uuid, name, email, password_hash, email_verified, 
                avatar, bio, created_at, updated_at
         FROM users WHERE email = ?1"
    )?;
    
    let user = stmt.query_row(params![email], |row| {
        Ok(User {
            id: row.get(0)?,
            uuid: row.get(1)?,
            name: row.get(2)?,
            email: row.get(3)?,
            password_hash: row.get(4)?,
            email_verified: row.get(5)?,
            avatar: row.get(6)?,
            bio: row.get(7)?,
            created_at: row.get(8)?,
            updated_at: row.get(9)?,
        })
    }).optional()?;
    
    Ok(user)
}

pub fn update_user_password(
    conn: &Connection,
    uuid: &str,
    password_hash: &str,
    updated_at: i64,
) -> Result<()> {
    conn.execute(
        "UPDATE users SET password_hash = ?1, updated_at = ?2 WHERE uuid = ?3",
        params![password_hash, updated_at, uuid],
    )?;
    Ok(())
}

// Session operations
pub fn create_session(
    conn: &Connection,
    id: &str,
    user_uuid: &str,
    created_at: i64,
    expires_at: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO sessions (id, user_uuid, created_at, expires_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![id, user_uuid, created_at, expires_at],
    )?;
    Ok(())
}

pub fn get_session(conn: &Connection, id: &str) -> Result<Option<Session>> {
    let mut stmt = conn.prepare(
        "SELECT id, user_uuid, created_at, expires_at
         FROM sessions WHERE id = ?1 AND expires_at > ?2"
    )?;
    
    let now = chrono::Utc::now().timestamp();
    let session = stmt.query_row(params![id, now], |row| {
        Ok(Session {
            id: row.get(0)?,
            user_uuid: row.get(1)?,
            created_at: row.get(2)?,
            expires_at: row.get(3)?,
        })
    }).optional()?;
    
    Ok(session)
}

pub fn delete_session(conn: &Connection, id: &str) -> Result<()> {
    conn.execute("DELETE FROM sessions WHERE id = ?1", params![id])?;
    Ok(())
}

// Token operations
pub fn create_email_verification_token(
    conn: &Connection,
    token: &str,
    user_uuid: &str,
    created_at: i64,
    expires_at: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO email_verification_tokens (token, user_uuid, created_at, expires_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![token, user_uuid, created_at, expires_at],
    )?;
    Ok(())
}

// ... similar functions for password reset tokens
```

### Step 5: Expose Host Functions

**tauri-app/src-tauri/src/host_functions/database.rs:**

```rust
use extism::{Function, UserData, Val};
use serde::{Deserialize, Serialize};
use crate::db::Database;

#[derive(Serialize, Deserialize)]
struct CreateUserInput {
    uuid: String,
    name: String,
    email: String,
    password_hash: String,
}

#[derive(Serialize, Deserialize)]
struct CreateUserOutput {
    id: i64,
    success: bool,
}

pub fn create_db_host_functions() -> Vec<Function> {
    vec![
        Function::new(
            "db_create_user",
            [Val::I64],
            [Val::I64],
            UserData::default(),
            |_plugin, inputs, outputs, _user_data| {
                // Parse input JSON
                let input_ptr = inputs[0].unwrap_i64();
                let input_json = /* read memory from plugin */;
                let input: CreateUserInput = serde_json::from_str(&input_json)?;
                
                // Get database connection (from user_data or app state)
                // Execute operation
                let result = /* create user */;
                
                // Serialize output
                let output = CreateUserOutput {
                    id: result,
                    success: true,
                };
                let output_json = serde_json::to_string(&output)?;
                
                // Write to plugin memory and return pointer
                outputs[0] = Val::I64(/* output pointer */);
                Ok(())
            }
        ),
        // ... more functions
    ]
}
```

### Step 6: Update Plugin Manager

**tauri-app/src-tauri/src/plugins/loader.rs:**

Add host functions when loading plugins:

```rust
pub fn load_with_host_functions(
    &self,
    host_functions: Vec<extism::Function>,
) -> Result<()> {
    let manifest = extism_manifest::Manifest::new([
        extism_manifest::Wasm::file(self.manifest.wasm_path(&self.plugins_dir))
    ]);
    
    let mut plugin = extism::Plugin::new(
        &manifest,
        host_functions, // Add host functions here
        false,
    )?;
    
    // ... rest of initialization
}
```

### Step 7: Initialize Database in Tauri App

**tauri-app/src-tauri/src/lib.rs:**

```rust
use db::Database;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // Initialize database
            let db_path = app.path().app_data_dir()?.join("app.db");
            let db = Database::new(db_path)?;
            
            // Run migrations
            db::migrations::run_migrations(db.connection())?;
            
            // Create host functions
            let host_functions = host_functions::database::create_db_host_functions();
            
            // Store database in app state
            app.manage(AppState {
                plugin_manager: Arc::new(RwLock::new(plugin_manager)),
                database: Arc::new(Mutex::new(db)),
                host_functions,
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // ... existing commands
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

## Testing

### Database Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_user() {
        let conn = Connection::open_in_memory().unwrap();
        migrations::run_migrations(&conn).unwrap();
        
        let uuid = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp();
        
        let user_id = operations::create_user(
            &conn,
            &uuid,
            "testuser",
            "test@example.com",
            "hash123",
            now,
        ).unwrap();
        
        assert!(user_id > 0);
        
        let user = operations::get_user_by_email(&conn, "test@example.com")
            .unwrap()
            .unwrap();
        
        assert_eq!(user.name, "testuser");
        assert_eq!(user.email, "test@example.com");
    }
}
```

### Plugin Integration Test

Create test plugin that uses database host functions:

```rust
// wasm-plugins/db-test/src/lib.rs
use extism_pdk::*;

#[host_fn]
extern "ExtismHost" {
    fn db_create_user(input: String) -> String;
}

#[plugin_fn]
pub fn test_db_access(Json(input): Json<serde_json::Value>) -> FnResult<Json<serde_json::Value>> {
    let create_input = serde_json::json!({
        "uuid": "test-uuid",
        "name": "test",
        "email": "test@test.com",
        "password_hash": "hash"
    });
    
    let result = unsafe {
        db_create_user(serde_json::to_string(&create_input).unwrap())
            .unwrap()
    };
    
    Ok(Json(serde_json::from_str(&result)?))
}
```

## Host Function Protocol

### Request Format
```json
{
  "function": "create_user",
  "params": {
    "uuid": "uuid-here",
    "name": "username",
    "email": "email@example.com",
    "password_hash": "hash"
  }
}
```

### Response Format
```json
{
  "success": true,
  "data": {
    "id": 123
  },
  "error": null
}
```

### Error Format
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "USER_EXISTS",
    "message": "User with this email already exists"
  }
}
```

## Performance Considerations

### Connection Pooling

For multiple concurrent plugins:

```rust
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;

pub struct DatabasePool {
    pool: Pool<SqliteConnectionManager>,
}

impl DatabasePool {
    pub fn new(db_path: PathBuf, pool_size: u32) -> Result<Self> {
        let manager = SqliteConnectionManager::file(db_path);
        let pool = Pool::builder()
            .max_size(pool_size)
            .build(manager)?;
        Ok(DatabasePool { pool })
    }
    
    pub fn get(&self) -> Result<PooledConnection<SqliteConnectionManager>> {
        Ok(self.pool.get()?)
    }
}
```

### Prepared Statements

Cache prepared statements for frequently used queries:

```rust
use std::collections::HashMap;
use rusqlite::Statement;

pub struct PreparedStatements<'conn> {
    statements: HashMap<String, Statement<'conn>>,
}
```

## Security Considerations

1. **SQL Injection**: Always use parameterized queries
2. **Access Control**: Validate plugin permissions before database access
3. **Data Sanitization**: Validate all inputs from plugins
4. **Transaction Isolation**: Use transactions for multi-step operations

## Error Handling

```rust
#[derive(Debug)]
pub enum DatabaseError {
    ConnectionError(String),
    QueryError(String),
    NotFound,
    AlreadyExists,
    InvalidInput(String),
}

impl From<rusqlite::Error> for DatabaseError {
    fn from(err: rusqlite::Error) -> Self {
        match err {
            rusqlite::Error::QueryReturnedNoRows => DatabaseError::NotFound,
            rusqlite::Error::SqliteFailure(err, msg) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    DatabaseError::AlreadyExists
                } else {
                    DatabaseError::QueryError(msg.unwrap_or_default())
                }
            }
            _ => DatabaseError::QueryError(err.to_string()),
        }
    }
}
```

## Next Steps

1. **Implement the database module** (Step 1-4)
2. **Create host functions** (Step 5)
3. **Update plugin manager** (Step 6)
4. **Test with template plugin** (create db-test plugin)
5. **Proceed to auth plugin** (Phase 3)

## References

- [rusqlite Documentation](https://docs.rs/rusqlite/)
- [Extism Host Functions](https://extism.org/docs/concepts/host-functions)
- Reference code: `reference-code/db/schema.ts`
- Reference code: `reference-code/server/services/authService.ts`

---

**Status**: Ready to implement
**Estimated Time**: 4-6 hours
**Dependencies**: None (all prerequisites met)
