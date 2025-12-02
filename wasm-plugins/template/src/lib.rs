use extism_pdk::*;
use serde::{Deserialize, Serialize};

/// Example input structure
#[derive(Serialize, Deserialize, Debug)]
pub struct ExampleInput {
    pub message: String,
    pub count: Option<u32>,
}

/// Example output structure
#[derive(Serialize, Deserialize, Debug)]
pub struct ExampleOutput {
    pub result: String,
    pub timestamp: u64,
}

/// Example error structure
#[derive(Serialize, Deserialize, Debug)]
pub struct PluginError {
    pub error: String,
    pub code: String,
}

/// Example host function call
/// Define host functions that the plugin can call back to the host
#[host_fn]
extern "ExtismHost" {
    fn get_current_time() -> u64;
    fn log_message(message: String);
}

/// Example plugin function - simple greeting
/// 
/// This demonstrates basic JSON input/output
#[plugin_fn]
pub fn greet(Json(input): Json<ExampleInput>) -> FnResult<Json<ExampleOutput>> {
    let message = format!("Hello, {}!", input.message);
    
    // Call host function (if available, otherwise use fallback)
    let timestamp = unsafe {
        get_current_time()
            .unwrap_or(0)
    };
    
    Ok(Json(ExampleOutput {
        result: message,
        timestamp,
    }))
}

/// Example plugin function - repeat message
/// 
/// This demonstrates handling optional parameters
#[plugin_fn]
pub fn repeat(Json(input): Json<ExampleInput>) -> FnResult<Json<ExampleOutput>> {
    let count = input.count.unwrap_or(1);
    let result = (0..count)
        .map(|_| input.message.clone())
        .collect::<Vec<_>>()
        .join(" ");
    
    // Log to host (if log_message host function is available)
    unsafe {
        let _ = log_message(format!("Repeated message {} times", count));
    }
    
    Ok(Json(ExampleOutput {
        result,
        timestamp: 0,
    }))
}

/// Example plugin function - validate input
/// 
/// This demonstrates error handling
#[plugin_fn]
pub fn validate(Json(input): Json<ExampleInput>) -> FnResult<Json<ExampleOutput>> {
    if input.message.is_empty() {
        return Err(WithReturnCode::new(
            Error::msg("Message cannot be empty"),
            1,
        ));
    }
    
    if let Some(count) = input.count {
        if count == 0 || count > 100 {
            return Err(WithReturnCode::new(
                Error::msg("Count must be between 1 and 100"),
                2,
            ));
        }
    }
    
    Ok(Json(ExampleOutput {
        result: format!("Valid input: {}", input.message),
        timestamp: 0,
    }))
}

/// Example plugin function - get plugin info
/// 
/// This demonstrates returning static metadata
#[plugin_fn]
pub fn get_info(Json(_): Json<serde_json::Value>) -> FnResult<Json<serde_json::Value>> {
    let info = serde_json::json!({
        "name": "Plugin Template",
        "version": "0.1.0",
        "description": "Template for creating Extism-compatible WASM plugins",
        "functions": [
            {
                "name": "greet",
                "description": "Generate a greeting message",
                "input": {
                    "message": "string",
                    "count": "optional<u32>"
                },
                "output": {
                    "result": "string",
                    "timestamp": "u64"
                }
            },
            {
                "name": "repeat",
                "description": "Repeat a message N times",
                "input": {
                    "message": "string",
                    "count": "optional<u32>"
                },
                "output": {
                    "result": "string",
                    "timestamp": "u64"
                }
            },
            {
                "name": "validate",
                "description": "Validate input parameters",
                "input": {
                    "message": "string",
                    "count": "optional<u32>"
                },
                "output": {
                    "result": "string",
                    "timestamp": "u64"
                }
            },
            {
                "name": "get_info",
                "description": "Get plugin metadata",
                "input": "any",
                "output": "object"
            }
        ]
    });
    
    Ok(Json(info))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_greet() {
        let input = ExampleInput {
            message: "World".to_string(),
            count: None,
        };
        
        // Note: This test won't actually call the plugin_fn,
        // it's just for demonstration
        assert_eq!(input.message, "World");
    }
}
