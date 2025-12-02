use extism_pdk::*;
use serde::{Deserialize, Serialize};
use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};

// ============================================================================
// Host Function Declarations
// ============================================================================

/// Utility host functions provided by the Tauri application
#[host_fn("extism:host/user")]
extern "ExtismHost" {
    /// Generate random bytes - returns JSON array string of bytes
    fn generate_random_bytes(length: i64) -> String;
    
    /// Get current timestamp in seconds
    fn get_timestamp() -> i64;
}

/// Database host functions provided by the Tauri application
#[host_fn("extism:host/user")]
extern "ExtismHost" {
    /// Create a new user in the database
    fn db_create_user(json_request: String) -> String;
    
    /// Get user by email address
    fn db_get_user_by_email(email: String) -> String;
    
    /// Get user by UUID
    fn db_get_user_by_uuid(uuid: String) -> String;
    
    /// Update user password hash
    fn db_update_user_password(json_request: String) -> String;
    
    /// Create a new session
    fn db_create_session(json_request: String) -> String;
    
    /// Get session by ID
    fn db_get_session(session_id: String) -> String;
    
    /// Delete a session
    fn db_delete_session(session_id: String) -> String;

    /// Create an audit log entry
    fn db_create_audit_log(json_request: String) -> String;
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Simple UUID generation using random bytes from host
fn generate_uuid() -> FnResult<String> {
    let json_bytes = unsafe { generate_random_bytes(16)? };
    let random_bytes: Vec<u8> = serde_json::from_str(&json_bytes)
        .map_err(|e| Error::msg(format!("Failed to parse random bytes: {}", e)))?;
    Ok(format!("{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        random_bytes[0], random_bytes[1], random_bytes[2], random_bytes[3],
        random_bytes[4], random_bytes[5],
        random_bytes[6], random_bytes[7],
        random_bytes[8], random_bytes[9],
        random_bytes[10], random_bytes[11], random_bytes[12], random_bytes[13], random_bytes[14], random_bytes[15]
    ))
}

// ============================================================================
// Request/Response Structures
// ============================================================================

#[derive(Deserialize)]
pub struct SignupRequest {
    pub name: String,
    pub email: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct SignupResponse {
    pub success: bool,
    pub user_uuid: Option<String>,
    pub message: String,
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct LoginResponse {
    pub success: bool,
    pub session_id: Option<String>,
    pub user: Option<UserInfo>,
    pub message: String,
}

#[derive(Serialize, Deserialize)]
pub struct UserInfo {
    pub uuid: String,
    pub name: String,
    pub email: String,
}

#[derive(Deserialize)]
pub struct VerifySessionRequest {
    pub session_id: String,
}

#[derive(Serialize)]
pub struct VerifySessionResponse {
    pub success: bool,
    pub valid: bool,
    pub user_uuid: Option<String>,
}

#[derive(Deserialize)]
pub struct LogoutRequest {
    pub session_id: String,
}

#[derive(Serialize)]
pub struct GenericResponse {
    pub success: bool,
    pub message: String,
}

// Database response structures
#[derive(Deserialize)]
struct DbResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct User {
    uuid: String,
    name: String,
    email: String,
    password_hash: String,
}

#[derive(Deserialize)]
struct Session {
    id: String,
    user_uuid: String,
    expires_at: i64,
}

// ============================================================================
// Plugin Functions
// ============================================================================

/// Sign up a new user
#[plugin_fn]
pub fn signup(Json(req): Json<SignupRequest>) -> FnResult<Json<SignupResponse>> {
    // Validate input
    if req.name.is_empty() || req.email.is_empty() || req.password.is_empty() {
        return Ok(Json(SignupResponse {
            success: false,
            user_uuid: None,
            message: "Name, email, and password are required".to_string(),
        }));
    }
    
    if req.password.len() < 8 {
        return Ok(Json(SignupResponse {
            success: false,
            user_uuid: None,
            message: "Password must be at least 8 characters".to_string(),
        }));
    }
    
    // Check if user already exists
    let existing_user = unsafe {
        match db_get_user_by_email(req.email.clone()) {
            Ok(response) => {
                let db_resp: DbResponse<User> = serde_json::from_str(&response)
                    .map_err(|e| Error::msg(format!("Failed to parse response: {}", e)))?;
                db_resp.data
            }
            Err(_) => None,
        }
    };
    
    if existing_user.is_some() {
        return Ok(Json(SignupResponse {
            success: false,
            user_uuid: None,
            message: "User with this email already exists".to_string(),
        }));
    }
    
    // Generate salt using random bytes from host (returns JSON array string)
    let json_salt = unsafe { generate_random_bytes(16)? };
    let salt_bytes: Vec<u8> = serde_json::from_str(&json_salt)
        .map_err(|e| Error::msg(format!("Failed to parse salt bytes: {}", e)))?;
    
    // Ensure we have exactly 16 bytes
    if salt_bytes.len() != 16 {
        return Err(Error::msg(format!("Invalid salt length: expected 16, got {}", salt_bytes.len())).into());
    }
    
    // Convert Vec<u8> to [u8; 16] for SaltString
    let mut salt_array = [0u8; 16];
    salt_array.copy_from_slice(&salt_bytes);
    
    let salt = SaltString::encode_b64(&salt_array)
        .map_err(|e| Error::msg(format!("Salt encoding error: {}", e)))?;
    
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(req.password.as_bytes(), &salt)
        .map_err(|e| Error::msg(format!("Password hashing failed: {}", e)))?
        .to_string();
    
    // Generate UUID for user
    let user_uuid = generate_uuid()?;
    let created_at = unsafe { get_timestamp()? };
    
    // Create user in database
    let create_request = serde_json::json!({
        "uuid": user_uuid,
        "name": req.name,
        "email": req.email,
        "password_hash": password_hash,
        "created_at": created_at,
    });
    
    let result = unsafe {
        db_create_user(create_request.to_string())
            .map_err(|e| Error::msg(format!("Database error: {}", e)))?
    };
    
    let db_resp: DbResponse<i64> = serde_json::from_str(&result)
        .map_err(|e| Error::msg(format!("Failed to parse response: {}", e)))?;
    
    if !db_resp.success {
        return Ok(Json(SignupResponse {
            success: false,
            user_uuid: None,
            message: db_resp.error.unwrap_or_else(|| "Failed to create user".to_string()),
        }));
    }
    
    // Create audit log for signup
    let audit_request = serde_json::json!({
        "user_uuid": user_uuid,
        "action": "user.signup",
        "resource_type": "user",
        "resource_id": user_uuid.clone(),
        "metadata": serde_json::json!({
            "name": req.name,
            "email": req.email,
        }).to_string(),
        "ip_address": None::<String>,
        "user_agent": None::<String>,
    });
    
    let _ = unsafe {
        db_create_audit_log(audit_request.to_string())
    };
    
    Ok(Json(SignupResponse {
        success: true,
        user_uuid: Some(user_uuid),
        message: "User created successfully".to_string(),
    }))
}

/// Log in a user
#[plugin_fn]
pub fn login(Json(req): Json<LoginRequest>) -> FnResult<Json<LoginResponse>> {
    // Get user by email
    let user = unsafe {
        match db_get_user_by_email(req.email.clone()) {
            Ok(response) => {
                let db_resp: DbResponse<User> = serde_json::from_str(&response)
                    .map_err(|e| Error::msg(format!("Failed to parse response: {}", e)))?;
                db_resp.data
            }
            Err(_) => None,
        }
    };
    
    let user = match user {
        Some(u) => u,
        None => {
            // Log failed login attempt (user not found)
            let audit_request = serde_json::json!({
                "user_uuid": None::<String>,
                "action": "user.login.failed",
                "resource_type": "auth",
                "resource_id": None::<String>,
                "metadata": serde_json::json!({
                    "email": req.email,
                    "reason": "user_not_found"
                }).to_string(),
                "ip_address": None::<String>,
                "user_agent": None::<String>,
            });
            let _ = unsafe {
                db_create_audit_log(audit_request.to_string())
            };
            
            return Ok(Json(LoginResponse {
                success: false,
                session_id: None,
                user: None,
                message: "Invalid email or password".to_string(),
            }));
        }
    };
    
    // Verify password
    let parsed_hash = PasswordHash::new(&user.password_hash)
        .map_err(|e| Error::msg(format!("Invalid password hash: {}", e)))?;
    
    let argon2 = Argon2::default();
    if argon2.verify_password(req.password.as_bytes(), &parsed_hash).is_err() {
        // Log failed login attempt (wrong password)
        let audit_request = serde_json::json!({
            "user_uuid": user.uuid.clone(),
            "action": "user.login.failed",
            "resource_type": "auth",
            "resource_id": None::<String>,
            "metadata": serde_json::json!({
                "email": req.email,
                "reason": "invalid_password"
            }).to_string(),
            "ip_address": None::<String>,
            "user_agent": None::<String>,
        });
        let _ = unsafe {
            db_create_audit_log(audit_request.to_string())
        };
        
        return Ok(Json(LoginResponse {
            success: false,
            session_id: None,
            user: None,
            message: "Invalid email or password".to_string(),
        }));
    }
    
    // Create session
    let session_id = generate_uuid()?;
    let created_at = unsafe { get_timestamp()? };
    let expires_at = created_at + (7 * 24 * 60 * 60); // 7 days from now
    
    let session_request = serde_json::json!({
        "id": session_id,
        "user_uuid": user.uuid,
        "created_at": created_at,
        "expires_at": expires_at,
    });
    
    let result = unsafe {
        db_create_session(session_request.to_string())
            .map_err(|e| Error::msg(format!("Failed to create session: {}", e)))?
    };
    
    let db_resp: DbResponse<bool> = serde_json::from_str(&result)
        .map_err(|e| Error::msg(format!("Failed to parse response: {}", e)))?;
    
    if !db_resp.success {
        return Ok(Json(LoginResponse {
            success: false,
            session_id: None,
            user: None,
            message: "Failed to create session".to_string(),
        }));
    }
    
    // Create audit log for successful login
    let audit_request = serde_json::json!({
        "user_uuid": user.uuid.clone(),
        "action": "user.login",
        "resource_type": "session",
        "resource_id": session_id.clone(),
        "metadata": serde_json::json!({
            "email": req.email,
        }).to_string(),
        "ip_address": None::<String>,
        "user_agent": None::<String>,
    });
    
    let _ = unsafe {
        db_create_audit_log(audit_request.to_string())
    };

    Ok(Json(LoginResponse {
        success: true,
        session_id: Some(session_id.clone()),
        user: Some(UserInfo {
            uuid: user.uuid,
            name: user.name,
            email: user.email,
        }),
        message: "Login successful".to_string(),
    }))
}

/// Verify a session
#[plugin_fn]
pub fn verify_session(Json(req): Json<VerifySessionRequest>) -> FnResult<Json<VerifySessionResponse>> {
    let session = unsafe {
        match db_get_session(req.session_id.clone()) {
            Ok(response) => {
                let db_resp: DbResponse<Session> = serde_json::from_str(&response)
                    .map_err(|e| Error::msg(format!("Failed to parse response: {}", e)))?;
                db_resp.data
            }
            Err(_) => None,
        }
    };
    
    let session = match session {
        Some(s) => s,
        None => {
            return Ok(Json(VerifySessionResponse {
                success: true,
                valid: false,
                user_uuid: None,
            }));
        }
    };
    
    // Check if session is expired
    let now = unsafe { get_timestamp()? };
    if session.expires_at < now {
        // Delete expired session
        let _ = unsafe { db_delete_session(req.session_id) };
        
        return Ok(Json(VerifySessionResponse {
            success: true,
            valid: false,
            user_uuid: None,
        }));
    }
    
    Ok(Json(VerifySessionResponse {
        success: true,
        valid: true,
        user_uuid: Some(session.user_uuid),
    }))
}

/// Log out a user
#[plugin_fn]
pub fn logout(Json(req): Json<LogoutRequest>) -> FnResult<Json<GenericResponse>> {
    // Get session to retrieve user_uuid for audit log
    let session = unsafe {
        match db_get_session(req.session_id.clone()) {
            Ok(response) => {
                let db_resp: DbResponse<Session> = serde_json::from_str(&response)
                    .map_err(|e| Error::msg(format!("Failed to parse response: {}", e)))?;
                db_resp.data
            }
            Err(_) => None,
        }
    };
    
    let result = unsafe {
        db_delete_session(req.session_id.clone())
            .map_err(|e| Error::msg(format!("Failed to delete session: {}", e)))?
    };
    
    let db_resp: DbResponse<bool> = serde_json::from_str(&result)
        .map_err(|e| Error::msg(format!("Failed to parse response: {}", e)))?;
    
    if !db_resp.success {
        return Ok(Json(GenericResponse {
            success: false,
            message: "Failed to log out".to_string(),
        }));
    }
    
    // Create audit log for logout
    if let Some(session) = session {
        let audit_request = serde_json::json!({
            "user_uuid": session.user_uuid,
            "action": "user.logout",
            "resource_type": "session",
            "resource_id": req.session_id,
            "metadata": None::<String>,
            "ip_address": None::<String>,
            "user_agent": None::<String>,
        });
        
        let _ = unsafe {
            db_create_audit_log(audit_request.to_string())
        };
    }
    
    Ok(Json(GenericResponse {
        success: true,
        message: "Logged out successfully".to_string(),
    }))
}

/// Get plugin info
#[plugin_fn]
pub fn get_info(Json(_): Json<serde_json::Value>) -> FnResult<Json<serde_json::Value>> {
    Ok(Json(serde_json::json!({
        "name": "Authentication Plugin",
        "version": "0.1.0",
        "description": "User authentication with database host functions",
        "functions": [
            {
                "name": "signup",
                "description": "Create a new user account"
            },
            {
                "name": "login",
                "description": "Authenticate user and create session"
            },
            {
                "name": "verify_session",
                "description": "Check if session is valid"
            },
            {
                "name": "logout",
                "description": "End user session"
            }
        ]
    })))
}
