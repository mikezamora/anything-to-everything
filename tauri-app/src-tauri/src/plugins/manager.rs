//! Plugin manager for discovering and managing plugins

use super::{PluginLoader, PluginManifest};
use crate::plugins::manifest::EntryPoint;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};
use reqwest;
use wasmparser::{Parser, Payload};

pub struct PluginManager {
    plugins_dir: PathBuf,
    plugins: Arc<RwLock<HashMap<String, PluginLoader>>>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new(plugins_dir: PathBuf) -> Result<Self> {
        if !plugins_dir.exists() {
            std::fs::create_dir_all(&plugins_dir)
                .context("Failed to create plugins directory")?;
        }
        
        Ok(PluginManager {
            plugins_dir,
            plugins: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Discover and load all plugins
    pub async fn discover_plugins(&self) -> Result<()> {
        info!("Discovering plugins in: {:?}", self.plugins_dir);
        
        let mut loaded_count = 0;
        
        // Read plugins directory
        let entries = std::fs::read_dir(&self.plugins_dir)
            .context("Failed to read plugins directory")?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                // Look for plugin.json in each subdirectory
                let manifest_path = path.join("plugin.json");
                if manifest_path.exists() {
                    match self.load_plugin_from_manifest(&manifest_path, &path).await {
                        Ok(_) => loaded_count += 1,
                        Err(e) => warn!("Failed to load plugin from {:?}: {}", path, e),
                    }
                }
            }
        }
        
        info!("✅ Loaded {} plugins", loaded_count);
        Ok(())
    }
    
    /// Load a plugin from its manifest file
    async fn load_plugin_from_manifest(
        &self,
        manifest_path: &Path,
        plugin_dir: &Path,
    ) -> Result<()> {
        let manifest = PluginManifest::load_from_file(manifest_path)?;
        let plugin_name = manifest.name.clone();
        
        let loader = PluginLoader::load(manifest, plugin_dir)?;
        
        let mut plugins = self.plugins.write().await;
        plugins.insert(plugin_name, loader);
        
        Ok(())
    }
    
    /// Install a plugin from a directory
    pub async fn install_plugin(&self, source: &Path) -> Result<()> {
        info!("Installing plugin from: {:?}", source);
        
        let manifest_path = source.join("plugin.json");
        if !manifest_path.exists() {
            anyhow::bail!("plugin.json not found in: {:?}", source);
        }
        
        let manifest = PluginManifest::load_from_file(&manifest_path)?;
        let dest_dir = self.plugins_dir.join(&manifest.name);
        
        // Copy plugin directory
        if dest_dir.exists() {
            std::fs::remove_dir_all(&dest_dir)?;
        }
        
        copy_dir_all(source, &dest_dir)?;
        
        // Load the plugin
        self.load_plugin_from_manifest(&dest_dir.join("plugin.json"), &dest_dir)
            .await?;
        
        Ok(())
    }
    
    /// Execute a plugin function
    pub async fn execute_plugin(
        &self,
        plugin_name: &str,
        function: &str,
        input: &[u8],
    ) -> Result<Vec<u8>> {
        let mut plugins = self.plugins.write().await;
        
        let plugin = plugins
            .get_mut(plugin_name)
            .context(format!("Plugin not found: {}", plugin_name))?;
        
        plugin.call(function, input)
    }
    
    /// List all loaded plugins
    pub async fn list_plugins(&self) -> Vec<PluginManifest> {
        let plugins = self.plugins.read().await;
        plugins
            .values()
            .map(|loader| loader.manifest().clone())
            .collect()
    }
    
    /// Get a specific plugin
    pub async fn get_plugin(&self, name: &str) -> Option<PluginManifest> {
        let plugins = self.plugins.read().await;
        plugins.get(name).map(|loader| loader.manifest().clone())
    }
    
    /// Extract exported functions from a WASM module
    fn extract_wasm_exports(wasm_bytes: &[u8]) -> Vec<String> {
        let mut exports = Vec::new();
        
        for payload in Parser::new(0).parse_all(wasm_bytes) {
            if let Ok(Payload::ExportSection(reader)) = payload {
                for export in reader {
                    if let Ok(export) = export {
                        if matches!(export.kind, wasmparser::ExternalKind::Func) {
                            exports.push(export.name.to_string());
                        }
                    }
                }
            }
        }
        
        exports
    }
    
    /// Install a plugin from a URL (WASM file or manifest URL)
    pub async fn install_plugin_from_url(&self, url: &str) -> Result<()> {
        info!("Installing plugin from URL: {}", url);
        
        // Download the content
        let response = reqwest::get(url)
            .await
            .context("Failed to fetch plugin from URL")?;
        
        let content = response
            .bytes()
            .await
            .context("Failed to download plugin content")?;
        
        // Determine if it's a WASM file or manifest
        let is_wasm = url.ends_with(".wasm");
        
        if is_wasm {
            // For WASM files, create a minimal manifest
            let plugin_name = url
                .rsplit('/')
                .next()
                .unwrap_or("remote-plugin")
                .trim_end_matches(".wasm");
            
            let dest_dir = self.plugins_dir.join(plugin_name);
            std::fs::create_dir_all(&dest_dir)?;
            
            // Save the WASM file
            let wasm_path = dest_dir.join("plugin.wasm");
            std::fs::write(&wasm_path, &content)?;
            
            // Extract exported functions from WASM
            let exported_functions = Self::extract_wasm_exports(&content);
            let entry_points: Vec<EntryPoint> = exported_functions
                .into_iter()
                .map(|func_name| EntryPoint {
                    name: func_name.clone(),
                    function: func_name.clone(),
                    description: format!("Exported function: {}", func_name),
                    input_format: "json".to_string(),
                    output_format: "json".to_string(),
                })
                .collect();
            
            // Create a basic manifest
            let manifest = PluginManifest {
                name: plugin_name.to_string(),
                version: "0.1.0".to_string(),
                description: format!("Plugin loaded from {}", url),
                author: Some("Remote".to_string()),
                plugin_type: "remote".to_string(),
                wasm_module: "plugin.wasm".to_string(),
                wasm_config: Default::default(),
                capabilities: vec![],
                entry_points,
                dependencies: Default::default(),
            };
            
            let manifest_path = dest_dir.join("plugin.json");
            let manifest_json = serde_json::to_string_pretty(&manifest)?;
            std::fs::write(&manifest_path, manifest_json)?;
            
            // Load the plugin
            self.load_plugin_from_manifest(&manifest_path, &dest_dir)
                .await?;
        } else {
            // Assume it's a manifest JSON
            let manifest: PluginManifest = serde_json::from_slice(&content)
                .context("Failed to parse plugin manifest from URL")?;
            
            let dest_dir = self.plugins_dir.join(&manifest.name);
            std::fs::create_dir_all(&dest_dir)?;
            
            // Save the manifest
            let manifest_path = dest_dir.join("plugin.json");
            std::fs::write(&manifest_path, &content)?;
            
            // If the manifest references a remote WASM URL, download it
            if manifest.wasm_module.starts_with("http://") || manifest.wasm_module.starts_with("https://") {
                let wasm_url = &manifest.wasm_module;
                let wasm_response = reqwest::get(wasm_url)
                    .await
                    .context("Failed to fetch WASM module")?;
                
                let wasm_content = wasm_response
                    .bytes()
                    .await
                    .context("Failed to download WASM module")?;
                
                // Save with a local filename
                let wasm_filename = wasm_url
                    .rsplit('/')
                    .next()
                    .unwrap_or("plugin.wasm");
                let wasm_path = dest_dir.join(wasm_filename);
                std::fs::write(&wasm_path, wasm_content)?;
                
                // Update manifest to use local file
                let mut local_manifest = manifest.clone();
                local_manifest.wasm_module = wasm_filename.to_string();
                let manifest_json = serde_json::to_string_pretty(&local_manifest)?;
                std::fs::write(&manifest_path, manifest_json)?;
            }
            
            // Load the plugin
            self.load_plugin_from_manifest(&manifest_path, &dest_dir)
                .await?;
        }
        
        info!("✅ Plugin installed successfully from URL");
        Ok(())
    }
}

/// Recursively copy a directory
fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        
        // Skip .git directories
        if entry.file_name() == ".git" {
            continue;
        }
        
        if ty.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    
    Ok(())
}
