use rusqlite::{Connection, Result, params, OptionalExtension};
use crate::db::schema::*;

// ============================================================================
// User Operations
// ============================================================================

/// Create a new user
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

/// Get user by email
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

/// Get user by UUID
pub fn get_user_by_uuid(conn: &Connection, uuid: &str) -> Result<Option<User>> {
    let mut stmt = conn.prepare(
        "SELECT id, uuid, name, email, password_hash, email_verified, 
                avatar, bio, created_at, updated_at
         FROM users WHERE uuid = ?1"
    )?;
    
    let user = stmt.query_row(params![uuid], |row| {
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

/// Get user by name
pub fn get_user_by_name(conn: &Connection, name: &str) -> Result<Option<User>> {
    let mut stmt = conn.prepare(
        "SELECT id, uuid, name, email, password_hash, email_verified, 
                avatar, bio, created_at, updated_at
         FROM users WHERE name = ?1"
    )?;
    
    let user = stmt.query_row(params![name], |row| {
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

/// Update user password
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

/// Update user email verification status
pub fn update_user_email_verified(
    conn: &Connection,
    uuid: &str,
    verified: bool,
) -> Result<()> {
    conn.execute(
        "UPDATE users SET email_verified = ?1, updated_at = strftime('%s', 'now') WHERE uuid = ?2",
        params![verified, uuid],
    )?;
    Ok(())
}

/// Update user profile
pub fn update_user_profile(
    conn: &Connection,
    uuid: &str,
    name: Option<&str>,
    bio: Option<&str>,
    avatar: Option<&str>,
) -> Result<()> {
    // Build query dynamically but execute with named parameters
    if name.is_some() && bio.is_some() && avatar.is_some() {
        conn.execute(
            "UPDATE users SET name = ?1, bio = ?2, avatar = ?3, updated_at = strftime('%s', 'now') WHERE uuid = ?4",
            params![name.unwrap(), bio.unwrap(), avatar.unwrap(), uuid],
        )?;
    } else if name.is_some() && bio.is_some() {
        conn.execute(
            "UPDATE users SET name = ?1, bio = ?2, updated_at = strftime('%s', 'now') WHERE uuid = ?3",
            params![name.unwrap(), bio.unwrap(), uuid],
        )?;
    } else if name.is_some() && avatar.is_some() {
        conn.execute(
            "UPDATE users SET name = ?1, avatar = ?2, updated_at = strftime('%s', 'now') WHERE uuid = ?3",
            params![name.unwrap(), avatar.unwrap(), uuid],
        )?;
    } else if bio.is_some() && avatar.is_some() {
        conn.execute(
            "UPDATE users SET bio = ?1, avatar = ?2, updated_at = strftime('%s', 'now') WHERE uuid = ?3",
            params![bio.unwrap(), avatar.unwrap(), uuid],
        )?;
    } else if let Some(n) = name {
        conn.execute(
            "UPDATE users SET name = ?1, updated_at = strftime('%s', 'now') WHERE uuid = ?2",
            params![n, uuid],
        )?;
    } else if let Some(b) = bio {
        conn.execute(
            "UPDATE users SET bio = ?1, updated_at = strftime('%s', 'now') WHERE uuid = ?2",
            params![b, uuid],
        )?;
    } else if let Some(a) = avatar {
        conn.execute(
            "UPDATE users SET avatar = ?1, updated_at = strftime('%s', 'now') WHERE uuid = ?2",
            params![a, uuid],
        )?;
    } else {
        // Nothing to update, just update timestamp
        conn.execute(
            "UPDATE users SET updated_at = strftime('%s', 'now') WHERE uuid = ?1",
            params![uuid],
        )?;
    }
    Ok(())
}

// ============================================================================
// Session Operations
// ============================================================================

/// Create a new session
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

/// Get session by ID (only if not expired)
pub fn get_session(conn: &Connection, id: &str) -> Result<Option<Session>> {
    let mut stmt = conn.prepare(
        "SELECT id, user_uuid, created_at, expires_at
         FROM sessions WHERE id = ?1 AND expires_at > strftime('%s', 'now')"
    )?;
    
    let session = stmt.query_row(params![id], |row| {
        Ok(Session {
            id: row.get(0)?,
            user_uuid: row.get(1)?,
            created_at: row.get(2)?,
            expires_at: row.get(3)?,
        })
    }).optional()?;
    
    Ok(session)
}

/// Delete session by ID
pub fn delete_session(conn: &Connection, id: &str) -> Result<()> {
    conn.execute("DELETE FROM sessions WHERE id = ?1", params![id])?;
    Ok(())
}

/// Delete all sessions for a user
pub fn delete_user_sessions(conn: &Connection, user_uuid: &str) -> Result<()> {
    conn.execute("DELETE FROM sessions WHERE user_uuid = ?1", params![user_uuid])?;
    Ok(())
}

/// Clean up expired sessions
pub fn cleanup_expired_sessions(conn: &Connection) -> Result<usize> {
    let deleted = conn.execute(
        "DELETE FROM sessions WHERE expires_at <= strftime('%s', 'now')",
        [],
    )?;
    Ok(deleted)
}

// ============================================================================
// Email Verification Token Operations
// ============================================================================

/// Create email verification token
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

/// Get email verification token (only if not expired)
pub fn get_email_verification_token(
    conn: &Connection,
    token: &str,
) -> Result<Option<EmailVerificationToken>> {
    let mut stmt = conn.prepare(
        "SELECT token, user_uuid, created_at, expires_at
         FROM email_verification_tokens 
         WHERE token = ?1 AND expires_at > strftime('%s', 'now')"
    )?;
    
    let token_record = stmt.query_row(params![token], |row| {
        Ok(EmailVerificationToken {
            token: row.get(0)?,
            user_uuid: row.get(1)?,
            created_at: row.get(2)?,
            expires_at: row.get(3)?,
        })
    }).optional()?;
    
    Ok(token_record)
}

/// Delete email verification token
pub fn delete_email_verification_token(conn: &Connection, token: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM email_verification_tokens WHERE token = ?1",
        params![token],
    )?;
    Ok(())
}

// ============================================================================
// Password Reset Token Operations
// ============================================================================

/// Create password reset token
pub fn create_password_reset_token(
    conn: &Connection,
    token: &str,
    user_uuid: &str,
    created_at: i64,
    expires_at: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO password_reset_tokens (token, user_uuid, created_at, expires_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![token, user_uuid, created_at, expires_at],
    )?;
    Ok(())
}

/// Get password reset token (only if not expired)
pub fn get_password_reset_token(
    conn: &Connection,
    token: &str,
) -> Result<Option<PasswordResetToken>> {
    let mut stmt = conn.prepare(
        "SELECT token, user_uuid, created_at, expires_at
         FROM password_reset_tokens 
         WHERE token = ?1 AND expires_at > strftime('%s', 'now')"
    )?;
    
    let token_record = stmt.query_row(params![token], |row| {
        Ok(PasswordResetToken {
            token: row.get(0)?,
            user_uuid: row.get(1)?,
            created_at: row.get(2)?,
            expires_at: row.get(3)?,
        })
    }).optional()?;
    
    Ok(token_record)
}

/// Delete password reset token
pub fn delete_password_reset_token(conn: &Connection, token: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM password_reset_tokens WHERE token = ?1",
        params![token],
    )?;
    Ok(())
}

/// Delete all password reset tokens for a user
pub fn delete_user_password_reset_tokens(conn: &Connection, user_uuid: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM password_reset_tokens WHERE user_uuid = ?1",
        params![user_uuid],
    )?;
    Ok(())
}

// ============================================================================
// Audit Log Operations
// ============================================================================

/// Create an audit log entry
pub fn create_audit_log(
    conn: &Connection,
    id: &str,
    user_uuid: &str,
    action: &str,
    resource_type: Option<&str>,
    resource_id: Option<&str>,
    metadata: Option<&str>,
    ip_address: Option<&str>,
    user_agent: Option<&str>,
    created_at: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO audit_logs (id, user_uuid, action, resource_type, resource_id, 
                                 metadata, ip_address, user_agent, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            id,
            user_uuid,
            action,
            resource_type,
            resource_id,
            metadata,
            ip_address,
            user_agent,
            created_at
        ],
    )?;
    Ok(())
}

/// Get audit logs for a user with pagination
pub fn get_user_audit_logs(
    conn: &Connection,
    user_uuid: &str,
    limit: i32,
    offset: i32,
) -> Result<Vec<AuditLog>> {
    let mut stmt = conn.prepare(
        "SELECT id, user_uuid, action, resource_type, resource_id, 
                metadata, ip_address, user_agent, created_at
         FROM audit_logs 
         WHERE user_uuid = ?1
         ORDER BY created_at DESC
         LIMIT ?2 OFFSET ?3"
    )?;
    
    let audit_logs = stmt.query_map(params![user_uuid, limit, offset], |row| {
        Ok(AuditLog {
            id: row.get(0)?,
            user_uuid: row.get(1)?,
            action: row.get(2)?,
            resource_type: row.get(3)?,
            resource_id: row.get(4)?,
            metadata: row.get(5)?,
            ip_address: row.get(6)?,
            user_agent: row.get(7)?,
            created_at: row.get(8)?,
        })
    })?
    .collect::<Result<Vec<_>>>()?;
    
    Ok(audit_logs)
}

/// Get audit logs with filters
pub fn get_audit_logs_filtered(
    conn: &Connection,
    user_uuid: Option<&str>,
    action: Option<&str>,
    resource_type: Option<&str>,
    start_time: Option<i64>,
    end_time: Option<i64>,
    limit: i32,
    offset: i32,
) -> Result<Vec<AuditLog>> {
    let mut query = String::from(
        "SELECT id, user_uuid, action, resource_type, resource_id, 
                metadata, ip_address, user_agent, created_at
         FROM audit_logs WHERE 1=1"
    );
    
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    
    if let Some(uuid) = user_uuid {
        query.push_str(" AND user_uuid = ?");
        params.push(Box::new(uuid.to_string()));
    }
    
    if let Some(act) = action {
        query.push_str(" AND action = ?");
        params.push(Box::new(act.to_string()));
    }
    
    if let Some(res_type) = resource_type {
        query.push_str(" AND resource_type = ?");
        params.push(Box::new(res_type.to_string()));
    }
    
    if let Some(start) = start_time {
        query.push_str(" AND created_at >= ?");
        params.push(Box::new(start));
    }
    
    if let Some(end) = end_time {
        query.push_str(" AND created_at <= ?");
        params.push(Box::new(end));
    }
    
    query.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");
    params.push(Box::new(limit));
    params.push(Box::new(offset));
    
    let mut stmt = conn.prepare(&query)?;
    let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    
    let audit_logs = stmt.query_map(param_refs.as_slice(), |row| {
        Ok(AuditLog {
            id: row.get(0)?,
            user_uuid: row.get(1)?,
            action: row.get(2)?,
            resource_type: row.get(3)?,
            resource_id: row.get(4)?,
            metadata: row.get(5)?,
            ip_address: row.get(6)?,
            user_agent: row.get(7)?,
            created_at: row.get(8)?,
        })
    })?
    .collect::<Result<Vec<_>>>()?;
    
    Ok(audit_logs)
}

/// Count total audit logs for a user
pub fn count_user_audit_logs(conn: &Connection, user_uuid: &str) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM audit_logs WHERE user_uuid = ?1",
        params![user_uuid],
        |row| row.get(0),
    )?;
    Ok(count)
}

/// Delete old audit logs (cleanup older than specified timestamp)
pub fn delete_old_audit_logs(conn: &Connection, older_than: i64) -> Result<usize> {
    let deleted = conn.execute(
        "DELETE FROM audit_logs WHERE created_at < ?1",
        params![older_than],
    )?;
    Ok(deleted)
}
