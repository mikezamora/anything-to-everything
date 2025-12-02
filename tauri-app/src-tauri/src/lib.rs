mod plugins;
mod commands;
pub mod db;  // Make public for testing
mod host_functions;
mod tick_manager;

use commands::*;
use plugins::PluginManager;
use db::Database;
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
            // Get app data directory
            let app_data_dir = app.path().app_data_dir()
                .expect("Failed to get app data directory");
            
            // Initialize database
            let db_path = app_data_dir.join("app.db");
            tracing::info!("Initializing database at: {:?}", db_path);
            let database = Database::new(db_path)
                .expect("Failed to create database");
            
            // Run migrations
            database.with_connection(|conn| {
                db::migrations::run_migrations(conn)
            }).expect("Failed to run database migrations");
            
            // Create plugin manager with database and host functions
            let plugins_dir = app_data_dir.join("plugins");
            let mut plugin_manager = PluginManager::new_with_database(plugins_dir, Arc::new(database.clone()))
                .expect("Failed to create plugin manager");
            
            // Discover and load plugins
            tauri::async_runtime::block_on(async {
                plugin_manager.discover_plugins().await
            }).expect("Failed to discover plugins");
            
            tracing::info!("Host functions registered and ready for use by plugins");

            // Initialize tick manager
            let tick_manager = tick_manager::TickManager::new(60); // 60 ticks per second
            tracing::info!("Tick manager initialized with 60 TPS");

            // Store in app state
            app.manage(AppState {
                plugin_manager: Arc::new(RwLock::new(plugin_manager)),
                database: Arc::new(database),
                tick_manager: Arc::new(RwLock::new(tick_manager)),
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
            db_test_connection,
            db_get_schema_version,
            tick_start,
            tick_stop,
            tick_get_status,
            tick_get_current_tick,
            tick_set_rate,
            tick_register_session,
            tick_unregister_session,
            tick_add_client,
            tick_remove_client,
            tick_get_session_info,
            tick_get_active_sessions,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
