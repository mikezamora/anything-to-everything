//! Tauri commands for plugin management

use crate::plugins::{PluginManager, PluginManifest};
use crate::db::Database;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;

use crate::tick_manager::TickManager;

pub struct AppState {
    pub plugin_manager: Arc<RwLock<PluginManager>>,
    pub database: Arc<Database>,
    pub tick_manager: Arc<RwLock<TickManager>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub plugin_type: String,
    pub capabilities: Vec<String>,
    pub entry_points: Vec<EntryPointInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EntryPointInfo {
    pub name: String,
    pub description: String,
    pub input_format: String,
    pub output_format: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteRequest {
    pub input: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteResponse {
    pub output: serde_json::Value,
}

impl From<PluginManifest> for PluginInfo {
    fn from(manifest: PluginManifest) -> Self {
        PluginInfo {
            name: manifest.name,
            version: manifest.version,
            description: manifest.description,
            plugin_type: manifest.plugin_type,
            capabilities: manifest.capabilities,
            entry_points: manifest
                .entry_points
                .into_iter()
                .map(|ep| EntryPointInfo {
                    name: ep.name,
                    description: ep.description,
                    input_format: ep.input_format,
                    output_format: ep.output_format,
                })
                .collect(),
        }
    }
}

#[tauri::command]
pub async fn list_plugins(state: State<'_, AppState>) -> Result<Vec<PluginInfo>, String> {
    let manager = state.plugin_manager.read().await;
    let plugins = manager.list_plugins().await;
    Ok(plugins.into_iter().map(PluginInfo::from).collect())
}

#[tauri::command]
pub async fn get_plugin_info(
    state: State<'_, AppState>,
    name: String,
) -> Result<PluginInfo, String> {
    let manager = state.plugin_manager.read().await;
    let plugin = manager
        .get_plugin(&name)
        .await
        .ok_or_else(|| format!("Plugin not found: {}", name))?;
    Ok(PluginInfo::from(plugin))
}

#[tauri::command]
pub async fn execute_plugin(
    state: State<'_, AppState>,
    plugin_name: String,
    function: String,
    input: serde_json::Value,
) -> Result<ExecuteResponse, String> {
    let input_bytes = serde_json::to_vec(&input).map_err(|e| e.to_string())?;

    let manager = state.plugin_manager.read().await;
    let output_bytes = manager
        .execute_plugin(&plugin_name, &function, &input_bytes)
        .await
        .map_err(|e| e.to_string())?;

    let output: serde_json::Value =
        serde_json::from_slice(&output_bytes).map_err(|e| e.to_string())?;

    Ok(ExecuteResponse { output })
}

#[tauri::command]
pub async fn install_plugin(
    state: State<'_, AppState>,
    path: String,
) -> Result<String, String> {
    let plugin_path = PathBuf::from(path);
    let manager = state.plugin_manager.read().await;
    manager
        .install_plugin(&plugin_path)
        .await
        .map_err(|e| e.to_string())?;
    Ok("Plugin installed successfully".to_string())
}

#[tauri::command]
pub async fn install_plugin_from_url(
    state: State<'_, AppState>,
    url: String,
) -> Result<String, String> {
    let manager = state.plugin_manager.read().await;
    manager
        .install_plugin_from_url(&url)
        .await
        .map_err(|e| e.to_string())?;
    Ok("Plugin installed successfully from URL".to_string())
}

#[tauri::command]
pub async fn discover_plugins(state: State<'_, AppState>) -> Result<usize, String> {
    let manager = state.plugin_manager.read().await;
    manager.discover_plugins().await.map_err(|e| e.to_string())?;
    let plugins = manager.list_plugins().await;
    Ok(plugins.len())
}

// ============================================================================
// Database Test Commands
// ============================================================================

#[tauri::command]
pub async fn db_test_connection(state: State<'_, AppState>) -> Result<String, String> {
    state.database.with_connection(|conn| {
        conn.query_row("SELECT 1", [], |row| {
            let val: i32 = row.get(0)?;
            Ok(val)
        })
    })
    .map_err(|e| e.to_string())?;
    
    Ok("Database connection successful".to_string())
}

#[tauri::command]
pub async fn db_get_schema_version(state: State<'_, AppState>) -> Result<i32, String> {
    state.database.with_connection(|conn| {
        conn.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
    })
    .map_err(|e| e.to_string())
}

// ============================================================================
// Tick Manager Commands
// ============================================================================

use crate::tick_manager::TickManagerStatus;

#[tauri::command]
pub async fn tick_start(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let mut manager = state.tick_manager.write().await;
    manager.start()?;
    
    // Start the tick loop in background
    let tick_manager_clone = state.tick_manager.clone();
    tauri::async_runtime::spawn(async move {
        crate::tick_manager::start_tick_loop(tick_manager_clone, app_handle).await;
    });
    
    Ok("Tick manager started".to_string())
}

#[tauri::command]
pub async fn tick_stop(state: State<'_, AppState>) -> Result<String, String> {
    let mut manager = state.tick_manager.write().await;
    manager.stop()?;
    Ok("Tick manager stopped".to_string())
}

#[tauri::command]
pub async fn tick_get_status(state: State<'_, AppState>) -> Result<TickManagerStatus, String> {
    let manager = state.tick_manager.read().await;
    Ok(manager.get_status())
}

#[tauri::command]
pub async fn tick_get_current_tick(state: State<'_, AppState>) -> Result<u64, String> {
    let manager = state.tick_manager.read().await;
    Ok(manager.get_current_tick())
}

#[tauri::command]
pub async fn tick_set_rate(state: State<'_, AppState>, rate: u32) -> Result<String, String> {
    let mut manager = state.tick_manager.write().await;
    manager.set_tick_rate(rate)?;
    Ok(format!("Tick rate set to {} ticks/second", rate))
}

#[tauri::command]
pub async fn tick_register_session(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<String, String> {
    let mut manager = state.tick_manager.write().await;
    manager.register_session(session_id.clone());
    Ok(format!("Session {} registered", session_id))
}

#[tauri::command]
pub async fn tick_unregister_session(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<String, String> {
    let mut manager = state.tick_manager.write().await;
    manager.unregister_session(&session_id);
    Ok(format!("Session {} unregistered", session_id))
}

#[tauri::command]
pub async fn tick_add_client(
    state: State<'_, AppState>,
    session_id: String,
    client_id: String,
) -> Result<String, String> {
    let mut manager = state.tick_manager.write().await;
    manager.add_client_to_session(session_id.clone(), client_id.clone());
    Ok(format!("Client {} added to session {}", client_id, session_id))
}

#[tauri::command]
pub async fn tick_remove_client(
    state: State<'_, AppState>,
    session_id: String,
    client_id: String,
) -> Result<String, String> {
    let mut manager = state.tick_manager.write().await;
    manager.remove_client_from_session(&session_id, &client_id);
    Ok(format!("Client {} removed from session {}", client_id, session_id))
}

#[tauri::command]
pub async fn tick_get_session_info(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<(u64, usize), String> {
    let manager = state.tick_manager.read().await;
    manager.get_session_info(&session_id)
        .ok_or_else(|| format!("Session {} not found", session_id))
}

#[tauri::command]
pub async fn tick_get_active_sessions(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let manager = state.tick_manager.read().await;
    Ok(manager.get_active_sessions())
}
