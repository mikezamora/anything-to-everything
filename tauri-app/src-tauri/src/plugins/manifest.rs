//! Plugin manifest definition

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use anyhow::{Context, Result};

/// Plugin manifest describing a WASM plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Plugin name (unique identifier)
    pub name: String,
    
    /// Plugin version
    pub version: String,
    
    /// Human-readable description
    pub description: String,
    
    /// Plugin author
    pub author: Option<String>,
    
    /// Plugin type (service, converter, processor, ui)
    pub plugin_type: String,
    
    /// Path to WASM module (relative to manifest)
    pub wasm_module: String,
    
    /// WASM runtime configuration
    #[serde(default)]
    pub wasm_config: WasmConfig,
    
    /// Plugin capabilities
    #[serde(default)]
    pub capabilities: Vec<String>,
    
    /// Entry points (exported functions)
    #[serde(default)]
    pub entry_points: Vec<EntryPoint>,
    
    /// Dependencies on other plugins
    #[serde(default)]
    pub dependencies: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WasmConfig {
    /// Allowed HTTP hosts
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    
    /// Allowed filesystem paths
    #[serde(default)]
    pub allowed_paths: HashMap<String, String>,
    
    /// Custom configuration key-value pairs
    #[serde(default)]
    pub config: HashMap<String, String>,
    
    /// Memory limit in pages (64KB per page)
    pub memory_max_pages: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryPoint {
    /// Function name as seen by users
    pub name: String,
    
    /// Actual WASM function to call
    pub function: String,
    
    /// Description of what this function does
    pub description: String,
    
    /// Expected input format (json, binary, text)
    #[serde(default)]
    pub input_format: String,
    
    /// Expected output format
    #[serde(default)]
    pub output_format: String,
}

impl PluginManifest {
    /// Load manifest from plugin.json file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read plugin manifest")?;
        
        serde_json::from_str(&content)
            .context("Failed to parse plugin manifest")
    }
    
    /// Validate the manifest
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            anyhow::bail!("Plugin name cannot be empty");
        }
        
        if self.version.is_empty() {
            anyhow::bail!("Plugin version cannot be empty");
        }
        
        if self.wasm_module.is_empty() {
            anyhow::bail!("WASM module path cannot be empty");
        }
        
        Ok(())
    }
    
    /// Get the full path to the WASM module
    pub fn wasm_path(&self, plugin_dir: &Path) -> std::path::PathBuf {
        plugin_dir.join(&self.wasm_module)
    }
}
