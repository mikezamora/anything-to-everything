use extism_pdk::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Host Function Declarations
// ============================================================================

/// Utility host functions
#[host_fn("extism:host/user")]
extern "ExtismHost" {
    fn get_timestamp() -> i64;
    fn get_timestamp_nanos() -> i64;
}

/// Database host functions
#[host_fn("extism:host/user")]
extern "ExtismHost" {
    fn db_create_audit_log(json_request: String) -> String;
    fn db_get_user_audit_logs(json_request: String) -> String;
    fn db_get_audit_logs_filtered(json_request: String) -> String;
    fn db_count_user_audit_logs(json_request: String) -> String;
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateAuditLogInput {
    pub user_uuid: String,
    pub action: String,
    pub resource_type: Option<String>,
    pub resource_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetAuditLogsInput {
    pub user_uuid: String,
    pub page: Option<i32>,
    pub limit: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetAuditLogsFilteredInput {
    pub user_uuid: Option<String>,
    pub action: Option<String>,
    pub resource_type: Option<String>,
    pub start_time: Option<i64>,
    pub end_time: Option<i64>,
    pub page: Option<i32>,
    pub limit: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuditLogsResponse {
    pub logs: Vec<AuditLog>,
    pub total: i64,
    pub page: i32,
    pub limit: i32,
    pub pages: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PluginResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> PluginResponse<T> {
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

#[derive(Debug, Serialize, Deserialize)]
struct HostResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

// ============================================================================
// Host Function Wrappers
// ============================================================================

fn call_db_create_audit_log(
    id: &str,
    user_uuid: &str,
    action: &str,
    resource_type: Option<&str>,
    resource_id: Option<&str>,
    metadata: Option<&str>,
    ip_address: Option<&str>,
    user_agent: Option<&str>,
    created_at: i64,
) -> Result<(), Error> {
    let request = serde_json::json!({
        "id": id,
        "user_uuid": user_uuid,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "metadata": metadata,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "created_at": created_at,
    });

    let request_str = serde_json::to_string(&request)?;
    let response_json = unsafe { db_create_audit_log(request_str)? };
    let response: HostResponse<()> = serde_json::from_str(&response_json)?;

    if !response.success {
        return Err(Error::msg(
            response
                .error
                .unwrap_or_else(|| "Unknown database error".to_string()),
        ));
    }

    Ok(())
}

fn call_db_get_user_audit_logs(
    user_uuid: &str,
    limit: i32,
    offset: i32,
) -> Result<Vec<AuditLog>, Error> {
    let request = serde_json::json!({
        "user_uuid": user_uuid,
        "limit": limit,
        "offset": offset,
    });

    let request_str = serde_json::to_string(&request)?;
    let response_json = unsafe { db_get_user_audit_logs(request_str)? };
    let response: HostResponse<Vec<AuditLog>> = serde_json::from_str(&response_json)?;

    if !response.success {
        return Err(Error::msg(
            response
                .error
                .unwrap_or_else(|| "Unknown database error".to_string()),
        ));
    }

    Ok(response.data.unwrap_or_default())
}

fn call_db_get_audit_logs_filtered(
    user_uuid: Option<&str>,
    action: Option<&str>,
    resource_type: Option<&str>,
    start_time: Option<i64>,
    end_time: Option<i64>,
    limit: i32,
    offset: i32,
) -> Result<Vec<AuditLog>, Error> {
    let request = serde_json::json!({
        "user_uuid": user_uuid,
        "action": action,
        "resource_type": resource_type,
        "start_time": start_time,
        "end_time": end_time,
        "limit": limit,
        "offset": offset,
    });

    let request_str = serde_json::to_string(&request)?;
    let response_json = unsafe { db_get_audit_logs_filtered(request_str)? };
    let response: HostResponse<Vec<AuditLog>> = serde_json::from_str(&response_json)?;

    if !response.success {
        return Err(Error::msg(
            response
                .error
                .unwrap_or_else(|| "Unknown database error".to_string()),
        ));
    }

    Ok(response.data.unwrap_or_default())
}

fn call_db_count_user_audit_logs(user_uuid: &str) -> Result<i64, Error> {
    let request = serde_json::json!({ "uuid": user_uuid });

    let request_str = serde_json::to_string(&request)?;
    let response_json = unsafe { db_count_user_audit_logs(request_str)? };
    let response: HostResponse<i64> = serde_json::from_str(&response_json)?;

    if !response.success {
        return Err(Error::msg(
            response
                .error
                .unwrap_or_else(|| "Unknown database error".to_string()),
        ));
    }

    Ok(response.data.unwrap_or(0))
}

// ============================================================================
// Utility Functions
// ============================================================================

fn generate_id() -> FnResult<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let timestamp = unsafe { get_timestamp_nanos()? };
    let mut hasher = DefaultHasher::new();
    timestamp.hash(&mut hasher);
    Ok(format!("audit_{:x}", hasher.finish()))
}

// ============================================================================
// Plugin Functions
// ============================================================================

/// Create an audit log entry
#[plugin_fn]
pub fn create_audit_log(input: String) -> FnResult<String> {
    let input: CreateAuditLogInput = serde_json::from_str(&input)?;

    let id = generate_id()?;
    let created_at = unsafe { get_timestamp()? };

    let metadata_str = input.metadata.map(|m| serde_json::to_string(&m).ok()).flatten();

    call_db_create_audit_log(
        &id,
        &input.user_uuid,
        &input.action,
        input.resource_type.as_deref(),
        input.resource_id.as_deref(),
        metadata_str.as_deref(),
        input.ip_address.as_deref(),
        input.user_agent.as_deref(),
        created_at,
    )?;

    let response = PluginResponse::success(AuditLog {
        id,
        user_uuid: input.user_uuid,
        action: input.action,
        resource_type: input.resource_type,
        resource_id: input.resource_id,
        metadata: metadata_str,
        ip_address: input.ip_address,
        user_agent: input.user_agent,
        created_at,
    });

    Ok(serde_json::to_string(&response)?)
}

/// Get audit logs for a user with pagination
#[plugin_fn]
pub fn get_user_audit_logs(input: String) -> FnResult<String> {
    let input: GetAuditLogsInput = serde_json::from_str(&input)?;

    let page = input.page.unwrap_or(1).max(1);
    let limit = input.limit.unwrap_or(50).clamp(1, 200);
    let offset = (page - 1) * limit;

    let logs = call_db_get_user_audit_logs(&input.user_uuid, limit, offset)?;
    let total = call_db_count_user_audit_logs(&input.user_uuid)?;
    let pages = (total as f64 / limit as f64).ceil() as i32;

    let response = PluginResponse::success(AuditLogsResponse {
        logs,
        total,
        page,
        limit,
        pages,
    });

    Ok(serde_json::to_string(&response)?)
}

/// Get audit logs with filters
#[plugin_fn]
pub fn get_audit_logs_filtered(input: String) -> FnResult<String> {
    let input: GetAuditLogsFilteredInput = serde_json::from_str(&input)?;

    let page = input.page.unwrap_or(1).max(1);
    let limit = input.limit.unwrap_or(50).clamp(1, 200);
    let offset = (page - 1) * limit;

    let logs = call_db_get_audit_logs_filtered(
        input.user_uuid.as_deref(),
        input.action.as_deref(),
        input.resource_type.as_deref(),
        input.start_time,
        input.end_time,
        limit,
        offset,
    )?;

    // Note: For filtered queries, we can't easily get the total count
    // without executing the query twice. For now, return 0 or estimate based on results.
    let total = logs.len() as i64;
    let pages = if logs.len() == limit as usize { page + 1 } else { page };

    let response = PluginResponse::success(AuditLogsResponse {
        logs,
        total,
        page,
        limit,
        pages,
    });

    Ok(serde_json::to_string(&response)?)
}
