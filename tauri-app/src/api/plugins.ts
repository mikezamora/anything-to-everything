/**
 * Plugin API - Fully typed interface for WASM plugin operations
 */

import { invoke } from "@tauri-apps/api/core";
import type { PluginInfo, ExecuteResponse } from "../types/plugin";

/**
 * List all available plugins
 */
export async function listPlugins(): Promise<PluginInfo[]> {
  return await invoke<PluginInfo[]>("list_plugins");
}

/**
 * Get detailed information about a specific plugin
 */
export async function getPluginInfo(name: string): Promise<PluginInfo> {
  return await invoke<PluginInfo>("get_plugin_info", { name });
}

/**
 * Execute a plugin function with typed input/output
 */
export async function executePlugin<TInput = any, TOutput = any>(
  pluginName: string,
  functionName: string,
  input: TInput
): Promise<TOutput> {
  const response = await invoke<ExecuteResponse>("execute_plugin", {
    pluginName,
    function: functionName,
    input,
  });
  return response.output as TOutput;
}

/**
 * Install a plugin from a local path
 */
export async function installPlugin(path: string): Promise<string> {
  return await invoke<string>("install_plugin", { path });
}

/**
 * Install a plugin from a URL (WASM file or manifest JSON)
 */
export async function installPluginFromUrl(url: string): Promise<string> {
  return await invoke<string>("install_plugin_from_url", { url });
}

/**
 * Discover and load all plugins from the plugins directory
 */
export async function discoverPlugins(): Promise<number> {
  return await invoke<number>("discover_plugins");
}

// ============================================================================
// Database Test Functions
// ============================================================================

/**
 * Test database connection
 */
export async function testDatabaseConnection(): Promise<string> {
  return await invoke<string>("db_test_connection");
}

/**
 * Get database schema version
 */
export async function getDatabaseSchemaVersion(): Promise<number> {
  return await invoke<number>("db_get_schema_version");
}
