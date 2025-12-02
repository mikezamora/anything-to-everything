//! Tauri commands for plugin management

use crate::plugins::{PluginManager, PluginManifest};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;

pub struct AppState {
    pub plugin_manager: Arc<RwLock<PluginManager>>,
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
