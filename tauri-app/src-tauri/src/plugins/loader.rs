//! Plugin loader using Extism runtime

use super::manifest::PluginManifest;
use anyhow::{Context, Result};
use extism::{Plugin, Manifest, Wasm};
use std::path::Path;
use tracing::{debug, info};

pub struct PluginLoader {
    manifest: PluginManifest,
    plugin: Plugin,
}

impl PluginLoader {
    /// Load a plugin from its manifest with host functions
    pub fn load_with_host_functions(
        plugin_manifest: PluginManifest,
        plugin_dir: &Path,
        host_fns: Vec<extism::Function>,
    ) -> Result<Self> {
        info!("Loading plugin: {} with {} host functions", plugin_manifest.name, host_fns.len());
        
        // Validate manifest
        plugin_manifest.validate()?;
        
        // Get WASM module path
        let wasm_path = plugin_manifest.wasm_path(plugin_dir);
        debug!("WASM module path: {:?}", wasm_path);
        
        if !wasm_path.exists() {
            anyhow::bail!("WASM module not found: {:?}", wasm_path);
        }
        
        // Build Extism manifest
        let mut manifest = Manifest::new([Wasm::file(&wasm_path)]);
        
        // Add configuration
        for (key, value) in &plugin_manifest.wasm_config.config {
            manifest = manifest.with_config_key(key, value);
        }
        
        // Add allowed hosts
        for host in &plugin_manifest.wasm_config.allowed_hosts {
            manifest = manifest.with_allowed_host(host);
        }
        
        // Add allowed paths
        for (guest, host) in &plugin_manifest.wasm_config.allowed_paths {
            manifest = manifest.with_allowed_path(guest.clone(), host);
        }
        
        // Create plugin with host functions
        let plugin = Plugin::new(&manifest, host_fns, true)
            .map_err(|e| anyhow::anyhow!("Failed to create Extism plugin for '{}' from {:?}: {:?}", plugin_manifest.name, wasm_path, e))?;
        
        info!("Successfully loaded plugin: {}", plugin_manifest.name);
        
        Ok(Self {
            manifest: plugin_manifest,
            plugin,
        })
    }

    /// Load a plugin from its manifest (without host functions)
    pub fn load(plugin_manifest: PluginManifest, plugin_dir: &Path) -> Result<Self> {
        info!("Loading plugin: {}", plugin_manifest.name);
        
        // Validate manifest
        plugin_manifest.validate()?;
        
        // Get WASM module path
        let wasm_path = plugin_manifest.wasm_path(plugin_dir);
        debug!("WASM module path: {:?}", wasm_path);
        
        if !wasm_path.exists() {
            anyhow::bail!("WASM module not found: {:?}", wasm_path);
        }
        
        // Build Extism manifest
        let mut manifest = Manifest::new([Wasm::file(&wasm_path)]);
        
        // Add configuration
        for (key, value) in &plugin_manifest.wasm_config.config {
            manifest = manifest.with_config_key(key, value);
        }
        
        // Add allowed hosts
        for host in &plugin_manifest.wasm_config.allowed_hosts {
            manifest = manifest.with_allowed_host(host);
        }
        
        // Add allowed paths
        for (guest, host) in &plugin_manifest.wasm_config.allowed_paths {
            manifest = manifest.with_allowed_path(guest.clone(), host);
        }
        
        // Create plugin
        let plugin = Plugin::new(&manifest, [], true)
            .context("Failed to create Extism plugin")?;
        
        info!("✅ Plugin loaded: {}", plugin_manifest.name);
        
        Ok(PluginLoader {
            manifest: plugin_manifest,
            plugin,
        })
    }
    
    /// Call a plugin function
    pub fn call(&mut self, function: &str, input: &[u8]) -> Result<Vec<u8>> {
        debug!(
            "Calling function '{}' on plugin '{}'",
            function, self.manifest.name
        );
        
        let result = self
            .plugin
            .call::<&[u8], &[u8]>(function, input)
            .context(format!("Failed to call plugin function: {}", function))?;
        
        Ok(result.to_vec())
    }
    
    /// Check if plugin has a function
    pub fn has_function(&self, function: &str) -> bool {
        self.plugin.function_exists(function)
    }
    
    /// Get plugin manifest
    pub fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }
}
