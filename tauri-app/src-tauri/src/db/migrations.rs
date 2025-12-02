use rusqlite::{Connection, Result};

/// Run all database migrations
pub fn run_migrations(conn: &Connection) -> Result<()> {
    // Create version table if it doesn't exist
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
    
    if current_version < 2 {
        migrate_v2(conn)?;
    }
    
    tracing::info!("Database migrations complete. Current version: {}", get_schema_version(conn)?);
    Ok(())
}

/// Get current schema version
fn get_schema_version(conn: &Connection) -> Result<i32> {
    let version: i32 = conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version",
        [],
        |row| row.get(0),
    )
    .unwrap_or(0);
    Ok(version)
}

/// Migration v1: Initial schema
fn migrate_v1(conn: &Connection) -> Result<()> {
    tracing::info!("Running migration v1: Initial schema");
    
    conn.execute_batch(
        "BEGIN;
        
        CREATE TABLE users (
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
        
        CREATE INDEX idx_users_uuid ON users(uuid);
        CREATE INDEX idx_users_email ON users(email);
        CREATE INDEX idx_users_name ON users(name);
        
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY (user_uuid) REFERENCES users(uuid) ON DELETE CASCADE
        );
        
        CREATE INDEX idx_sessions_user_uuid ON sessions(user_uuid);
        CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
        
        CREATE TABLE email_verification_tokens (
            token TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY (user_uuid) REFERENCES users(uuid) ON DELETE CASCADE
        );
        
        CREATE INDEX idx_email_tokens_user_uuid ON email_verification_tokens(user_uuid);
        CREATE INDEX idx_email_tokens_expires_at ON email_verification_tokens(expires_at);
        
        CREATE TABLE password_reset_tokens (
            token TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY (user_uuid) REFERENCES users(uuid) ON DELETE CASCADE
        );
        
        CREATE INDEX idx_password_tokens_user_uuid ON password_reset_tokens(user_uuid);
        CREATE INDEX idx_password_tokens_expires_at ON password_reset_tokens(expires_at);
        
        INSERT INTO schema_version (version, applied_at) 
        VALUES (1, strftime('%s', 'now'));
        
        COMMIT;"
    )?;
    
    tracing::info!("Migration v1 complete");
    Ok(())
}

/// Migration v2: Audit logs
fn migrate_v2(conn: &Connection) -> Result<()> {
    tracing::info!("Running migration v2: Audit logs");
    
    conn.execute_batch(
        "BEGIN;
        
        CREATE TABLE audit_logs (
            id TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id TEXT,
            metadata TEXT,
            ip_address TEXT,
            user_agent TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (user_uuid) REFERENCES users(uuid) ON DELETE CASCADE
        );
        
        CREATE INDEX idx_audit_user_uuid ON audit_logs(user_uuid);
        CREATE INDEX idx_audit_action ON audit_logs(action);
        CREATE INDEX idx_audit_created_at ON audit_logs(created_at);
        CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);
        
        INSERT INTO schema_version (version, applied_at) 
        VALUES (2, strftime('%s', 'now'));
        
        COMMIT;"
    )?;
    
    tracing::info!("Migration v2 complete");
    Ok(())
}
