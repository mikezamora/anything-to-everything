/**
 * Plugin types for the WASM plugin system
 */

export interface PluginInfo {
  name: string;
  version: string;
  description: string;
  plugin_type: string;
  capabilities: string[];
  entry_points: EntryPointInfo[];
}

export interface EntryPointInfo {
  name: string;
  description: string;
  input_format: string;
  output_format: string;
}

export interface ExecuteResponse {
  output: any;
}

export interface PluginError {
  error: string;
}
