use extism::{host_fn, Function, UserData, PTR};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::HostFunctionState;
use crate::db::{operations, schema::*};

/// Request types
#[derive(Deserialize, Serialize)]
struct CreateUserRequest {
    uuid: String,
    name: String,
    email: String,
    password_hash: String,
    created_at: i64,
}

#[derive(Deserialize, Serialize)]
struct UpdateUserProfileRequest {
    uuid: String,
    name: Option<String>,
    email: Option<String>,
    avatar: Option<String>,
    bio: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct UpdatePasswordRequest {
    uuid: String,
    password_hash: String,
    updated_at: i64,
}

#[derive(Deserialize, Serialize)]
struct UpdateEmailVerifiedRequest {
    uuid: String,
    verified: bool,
}

#[derive(Deserialize, Serialize)]
struct CreateSessionRequest {
    id: String,
    user_uuid: String,
    created_at: i64,
    expires_at: i64,
}

#[derive(Deserialize, Serialize)]
struct CreateEmailVerificationTokenRequest {
    token: String,
    user_uuid: String,
    created_at: i64,
    expires_at: i64,
}

#[derive(Deserialize, Serialize)]
struct CreatePasswordResetTokenRequest {
    token: String,
    user_uuid: String,
    created_at: i64,
    expires_at: i64,
}

#[derive(Deserialize, Serialize)]
struct GetUserRequest {
    uuid: String,
}

#[derive(Deserialize, Serialize)]
struct TokenRequest {
    token: String,
}

/// Generic response
#[derive(Serialize, Deserialize)]
struct HostResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

impl<T> HostResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
        }
    }
}

// Define host functions using Extism 1.13 host_fn! macro
host_fn!(db_create_user(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: CreateUserRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<i64>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::create_user(conn, &request.uuid, &request.name, &request.email, &request.password_hash, request.created_at)
    });

    let response = match result {
        Ok(id) => HostResponse::success(id),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

host_fn!(db_get_user_by_email(user_data: Arc<HostFunctionState>; email: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let result = state.database.with_connection(|conn| operations::get_user_by_email(conn, &email));
    let response = match result {
        Ok(user) => HostResponse::success(user),
        Err(e) => HostResponse::error(e.to_string()),
    };
    Ok(serde_json::to_string(&response).unwrap_or_default())
});

host_fn!(db_get_user_by_uuid(user_data: Arc<HostFunctionState>; uuid: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let result = state.database.with_connection(|conn| operations::get_user_by_uuid(conn, &uuid));
    let response = match result {
        Ok(user) => HostResponse::success(user),
        Err(e) => HostResponse::error(e.to_string()),
    };
    Ok(serde_json::to_string(&response).unwrap_or_default())
});

host_fn!(db_update_user_password(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: UpdatePasswordRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<bool>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::update_user_password(conn, &request.uuid, &request.password_hash, request.updated_at)
    });

    let response = match result {
        Ok(_) => HostResponse::success(true),
        Err(e) => HostResponse::error(e.to_string()),
    };
    Ok(serde_json::to_string(&response).unwrap_or_default())
});

host_fn!(db_create_session(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: CreateSessionRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<bool>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::create_session(conn, &request.id, &request.user_uuid, request.created_at, request.expires_at)
    });

    let response = match result {
        Ok(_) => HostResponse::success(true),
        Err(e) => HostResponse::error(e.to_string()),
    };
    Ok(serde_json::to_string(&response).unwrap_or_default())
});

host_fn!(db_get_session(user_data: Arc<HostFunctionState>; session_id: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let result = state.database.with_connection(|conn| operations::get_session(conn, &session_id));
    let response = match result {
        Ok(session) => HostResponse::success(session),
        Err(e) => HostResponse::error(e.to_string()),
    };
    Ok(serde_json::to_string(&response).unwrap_or_default())
});

host_fn!(db_delete_session(user_data: Arc<HostFunctionState>; session_id: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let result = state.database.with_connection(|conn| operations::delete_session(conn, &session_id));
    let response = match result {
        Ok(_) => HostResponse::success(true),
        Err(e) => HostResponse::error(e.to_string()),
    };
    Ok(serde_json::to_string(&response).unwrap_or_default())
});

// Public functions to create Function objects from host_fn definitions

pub fn create_user_host(state: Arc<HostFunctionState>) -> Function {
    Function::new(
        "db_create_user",
        [PTR],
        [PTR],
        UserData::new(state),
        db_create_user,
    )
}

pub fn get_user_by_email_host(state: Arc<HostFunctionState>) -> Function {
    Function::new(
        "db_get_user_by_email",
        [PTR],
        [PTR],
        UserData::new(state),
        db_get_user_by_email,
    )
}

pub fn get_user_by_uuid_host(state: Arc<HostFunctionState>) -> Function {
    Function::new(
        "db_get_user_by_uuid",
        [PTR],
        [PTR],
        UserData::new(state),
        db_get_user_by_uuid,
    )
}

pub fn update_user_password_host(state: Arc<HostFunctionState>) -> Function {
    Function::new(
        "db_update_user_password",
        [PTR],
        [PTR],
        UserData::new(state),
        db_update_user_password,
    )
}

pub fn create_session_host(state: Arc<HostFunctionState>) -> Function {
    Function::new(
        "db_create_session",
        [PTR],
        [PTR],
        UserData::new(state),
        db_create_session,
    )
}

pub fn get_session_host(state: Arc<HostFunctionState>) -> Function {
    Function::new(
        "db_get_session",
        [PTR],
        [PTR],
        UserData::new(state),
        db_get_session,
    )
}

pub fn delete_session_host(state: Arc<HostFunctionState>) -> Function {
    Function::new(
        "db_delete_session",
        [PTR],
        [PTR],
        UserData::new(state),
        db_delete_session,
    )
}

// Stub implementations for remaining host functions
// These will be properly implemented with the correct host_fn! definitions

host_fn!(db_update_user_email_verified(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: UpdateEmailVerifiedRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<()>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::update_user_email_verified(conn, &request.uuid, request.verified)
    });

    let response = match result {
        Ok(_) => HostResponse::success(()),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn update_user_email_verified_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_update_user_email_verified", [PTR], [PTR], UserData::new(state), db_update_user_email_verified)
}

host_fn!(db_update_user_profile(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: UpdateUserProfileRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<()>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::update_user_profile(
            conn, 
            &request.uuid, 
            request.name.as_deref(), 
            request.bio.as_deref(), 
            request.avatar.as_deref()
        )
    });

    let response = match result {
        Ok(_) => HostResponse::success(()),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn update_user_profile_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_update_user_profile", [PTR], [PTR], UserData::new(state), db_update_user_profile)
}

host_fn!(db_delete_user_sessions(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: GetUserRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<()>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::delete_user_sessions(conn, &request.uuid)
    });

    let response = match result {
        Ok(_) => HostResponse::success(()),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn delete_user_sessions_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_delete_user_sessions", [PTR], [PTR], UserData::new(state), db_delete_user_sessions)
}

pub fn cleanup_expired_sessions_host(state: Arc<HostFunctionState>) -> Function {
    host_fn!(stub_cleanup_sessions(user_data: Arc<HostFunctionState>;) -> String {
        let state = user_data.get()?;
        let state = state.lock().unwrap();
        let result = state.database.with_connection(|conn| operations::cleanup_expired_sessions(conn));
        let response = match result {
            Ok(count) => HostResponse::success(count),
            Err(e) => HostResponse::error(e.to_string()),
        };
        Ok(serde_json::to_string(&response).unwrap_or_default())
    });
    Function::new("db_cleanup_expired_sessions", [PTR], [PTR], UserData::new(state), stub_cleanup_sessions)
}

host_fn!(db_create_email_verification_token(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: CreateEmailVerificationTokenRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<String>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::create_email_verification_token(conn, &request.user_uuid, &request.token, request.created_at, request.expires_at)
    });

    let response = match result {
        Ok(token) => HostResponse::success(token),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn create_email_verification_token_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_create_email_verification_token", [PTR], [PTR], UserData::new(state), db_create_email_verification_token)
}

host_fn!(db_get_email_verification_token(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: TokenRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<Option<EmailVerificationToken>>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::get_email_verification_token(conn, &request.token)
    });

    let response = match result {
        Ok(token) => HostResponse::success(token),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn get_email_verification_token_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_get_email_verification_token", [PTR], [PTR], UserData::new(state), db_get_email_verification_token)
}

host_fn!(db_delete_email_verification_token(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: TokenRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<()>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::delete_email_verification_token(conn, &request.token)
    });

    let response = match result {
        Ok(_) => HostResponse::success(()),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn delete_email_verification_token_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_delete_email_verification_token", [PTR], [PTR], UserData::new(state), db_delete_email_verification_token)
}

host_fn!(db_create_password_reset_token(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: CreatePasswordResetTokenRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<String>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::create_password_reset_token(conn, &request.user_uuid, &request.token, request.created_at, request.expires_at)
    });

    let response = match result {
        Ok(token) => HostResponse::success(token),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn create_password_reset_token_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_create_password_reset_token", [PTR], [PTR], UserData::new(state), db_create_password_reset_token)
}

host_fn!(db_get_password_reset_token(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: TokenRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<Option<PasswordResetToken>>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::get_password_reset_token(conn, &request.token)
    });

    let response = match result {
        Ok(token) => HostResponse::success(token),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn get_password_reset_token_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_get_password_reset_token", [PTR], [PTR], UserData::new(state), db_get_password_reset_token)
}

host_fn!(db_delete_password_reset_token(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: TokenRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<()>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::delete_password_reset_token(conn, &request.token)
    });

    let response = match result {
        Ok(_) => HostResponse::success(()),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn delete_password_reset_token_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_delete_password_reset_token", [PTR], [PTR], UserData::new(state), db_delete_password_reset_token)
}

host_fn!(db_delete_user_password_reset_tokens(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: GetUserRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<()>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::delete_user_password_reset_tokens(conn, &request.uuid)
    });

    let response = match result {
        Ok(_) => HostResponse::success(()),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn delete_user_password_reset_tokens_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_delete_user_password_reset_tokens", [PTR], [PTR], UserData::new(state), db_delete_user_password_reset_tokens)
}

// ============================================================================
// Audit Log Host Functions
// ============================================================================

#[derive(Deserialize, Serialize)]
struct CreateAuditLogRequest {
    id: String,
    user_uuid: String,
    action: String,
    resource_type: Option<String>,
    resource_id: Option<String>,
    metadata: Option<String>,
    ip_address: Option<String>,
    user_agent: Option<String>,
    created_at: i64,
}

#[derive(Deserialize, Serialize)]
struct GetAuditLogsRequest {
    user_uuid: String,
    limit: i32,
    offset: i32,
}

#[derive(Deserialize, Serialize)]
struct GetAuditLogsFilteredRequest {
    user_uuid: Option<String>,
    action: Option<String>,
    resource_type: Option<String>,
    start_time: Option<i64>,
    end_time: Option<i64>,
    limit: i32,
    offset: i32,
}

host_fn!(db_create_audit_log(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: CreateAuditLogRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<()>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::create_audit_log(
            conn,
            &request.id,
            &request.user_uuid,
            &request.action,
            request.resource_type.as_deref(),
            request.resource_id.as_deref(),
            request.metadata.as_deref(),
            request.ip_address.as_deref(),
            request.user_agent.as_deref(),
            request.created_at,
        )
    });

    let response = match result {
        Ok(_) => HostResponse::success(()),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn create_audit_log_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_create_audit_log", [PTR], [PTR], UserData::new(state), db_create_audit_log)
}

host_fn!(db_get_user_audit_logs(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: GetAuditLogsRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<Vec<AuditLog>>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::get_user_audit_logs(conn, &request.user_uuid, request.limit, request.offset)
    });

    let response = match result {
        Ok(logs) => HostResponse::success(logs),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn get_user_audit_logs_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_get_user_audit_logs", [PTR], [PTR], UserData::new(state), db_get_user_audit_logs)
}

host_fn!(db_get_audit_logs_filtered(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: GetAuditLogsFilteredRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<Vec<AuditLog>>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::get_audit_logs_filtered(
            conn,
            request.user_uuid.as_deref(),
            request.action.as_deref(),
            request.resource_type.as_deref(),
            request.start_time,
            request.end_time,
            request.limit,
            request.offset,
        )
    });

    let response = match result {
        Ok(logs) => HostResponse::success(logs),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn get_audit_logs_filtered_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_get_audit_logs_filtered", [PTR], [PTR], UserData::new(state), db_get_audit_logs_filtered)
}

host_fn!(db_count_user_audit_logs(user_data: Arc<HostFunctionState>; input: String) -> String {
    let state = user_data.get()?;
    let state = state.lock().unwrap();
    let request: GetUserRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = HostResponse::<i64>::error(format!("JSON parse error: {}", e));
            return Ok(serde_json::to_string(&resp).unwrap_or_default());
        }
    };

    let result = state.database.with_connection(|conn| {
        operations::count_user_audit_logs(conn, &request.uuid)
    });

    let response = match result {
        Ok(count) => HostResponse::success(count),
        Err(e) => HostResponse::error(e.to_string()),
    };

    Ok(serde_json::to_string(&response).unwrap_or_default())
});

pub fn count_user_audit_logs_host(state: Arc<HostFunctionState>) -> Function {
    Function::new("db_count_user_audit_logs", [PTR], [PTR], UserData::new(state), db_count_user_audit_logs)
}