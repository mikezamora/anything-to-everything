mod plugins;
mod commands;

use commands::*;
use plugins::PluginManager;
use std::sync::Arc;
use tauri::Manager;
use tokio::sync::RwLock;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Get app data directory for plugins
            let app_data_dir = app.path().app_data_dir()
                .expect("Failed to get app data directory");
            let plugins_dir = app_data_dir.join("plugins");

            // Create plugin manager
            let plugin_manager = PluginManager::new(plugins_dir)
                .expect("Failed to create plugin manager");

            // Store in app state
            app.manage(AppState {
                plugin_manager: Arc::new(RwLock::new(plugin_manager)),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_plugins,
            get_plugin_info,
            execute_plugin,
            install_plugin,
            install_plugin_from_url,
            discover_plugins,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
