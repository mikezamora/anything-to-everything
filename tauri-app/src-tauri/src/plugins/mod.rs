//! Plugin system for loading and managing WASM plugins

mod manifest;
mod manager;
mod loader;

pub use manifest::PluginManifest;
pub use manager::PluginManager;
pub use loader::PluginLoader;
