use serde::{Deserialize, Serialize};

/// User record
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Session record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub user_uuid: String,
    pub created_at: i64,
    pub expires_at: i64,
}

/// Email verification token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailVerificationToken {
    pub token: String,
    pub user_uuid: String,
    pub created_at: i64,
    pub expires_at: i64,
}

/// Password reset token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordResetToken {
    pub token: String,
    pub user_uuid: String,
    pub created_at: i64,
    pub expires_at: i64,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub id: String,
    pub user_uuid: String,
    pub action: String,
    pub resource_type: Option<String>,
    pub resource_id: Option<String>,
    pub metadata: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub created_at: i64,
}
