pub mod database;

use extism::{Function, UserData, CurrentPlugin, Val, ValType, PTR};
use std::sync::Arc;

use crate::db::Database;

/// User data passed to host functions containing app state
pub struct HostFunctionState {
    pub database: Arc<Database>,
}

// Generate random bytes host function using host_fn! macro - returns JSON array string
extism::host_fn!(generate_random_bytes_impl(user_data: (); length: i64) -> String {
    use rand::RngCore;
    let length = length as usize;
    tracing::info!("Generating {} random bytes", length);
    let mut random_bytes = vec![0u8; length];
    rand::thread_rng().fill_bytes(&mut random_bytes);
    tracing::info!("Generated {} bytes: {:?}", random_bytes.len(), &random_bytes[..random_bytes.len().min(8)]);
    // Return as JSON array string
    Ok(serde_json::to_string(&random_bytes).unwrap_or_default())
});

pub fn generate_random_bytes_host() -> Function {
    Function::new("generate_random_bytes", [PTR], [PTR], UserData::new(()), generate_random_bytes_impl)
}

// Get current timestamp in seconds host function
pub fn get_timestamp_host() -> Function {
    Function::new(
        "get_timestamp",
        [],
        [ValType::I64],
        UserData::new(()),
        |_plugin: &mut CurrentPlugin, _inputs: &[Val], outputs: &mut [Val], _user_data: UserData<()>| {
            use std::time::{SystemTime, UNIX_EPOCH};
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            outputs[0] = Val::I64(timestamp);
            Ok(())
        },
    )
}

// Get current timestamp in nanoseconds host function
pub fn get_timestamp_nanos_host() -> Function {
    Function::new(
        "get_timestamp_nanos",
        [],
        [ValType::I64],
        UserData::new(()),
        |_plugin: &mut CurrentPlugin, _inputs: &[Val], outputs: &mut [Val], _user_data: UserData<()>| {
            use std::time::{SystemTime, UNIX_EPOCH};
            let timestamp_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as i64;
            outputs[0] = Val::I64(timestamp_nanos);
            Ok(())
        },
    )
}

/// Register all host functions with the Extism plugin
pub fn register_host_functions(database: Arc<Database>) -> Vec<Function> {
    let state = Arc::new(HostFunctionState { database });
    
    vec![
        // Utility functions - use () as user_data since they don't need database state
        generate_random_bytes_host(),
        get_timestamp_host(),
        get_timestamp_nanos_host(),
        
        // User operations
        database::create_user_host(state.clone()),
        database::get_user_by_email_host(state.clone()),
        database::get_user_by_uuid_host(state.clone()),
        database::update_user_password_host(state.clone()),
        database::update_user_email_verified_host(state.clone()),
        database::update_user_profile_host(state.clone()),
        
        // Session operations
        database::create_session_host(state.clone()),
        database::get_session_host(state.clone()),
        database::delete_session_host(state.clone()),
        database::delete_user_sessions_host(state.clone()),
        database::cleanup_expired_sessions_host(state.clone()),
        
        // Email verification token operations
        database::create_email_verification_token_host(state.clone()),
        database::get_email_verification_token_host(state.clone()),
        database::delete_email_verification_token_host(state.clone()),
        
        // Password reset token operations
        database::create_password_reset_token_host(state.clone()),
        database::get_password_reset_token_host(state.clone()),
        database::delete_password_reset_token_host(state.clone()),
        database::delete_user_password_reset_tokens_host(state.clone()),
        
        // Audit log operations
        database::create_audit_log_host(state.clone()),
        database::get_user_audit_logs_host(state.clone()),
        database::get_audit_logs_filtered_host(state.clone()),
        database::count_user_audit_logs_host(state.clone()),
    ]
}
