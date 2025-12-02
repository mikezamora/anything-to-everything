/**
 * React hook for managing plugin operations
 */

import { useState, useEffect, useCallback } from "react";
import {
  listPlugins,
  getPluginInfo,
  executePlugin,
  installPlugin,
  installPluginFromUrl,
  discoverPlugins,
} from "../api/plugins";
import type { PluginInfo } from "../types/plugin";

export function usePlugins() {
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadPlugins = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const pluginList = await listPlugins();
      setPlugins(pluginList);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load plugins");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPlugins();
  }, [loadPlugins]);

  return {
    plugins,
    loading,
    error,
    loadPlugins,
  };
}

export function usePluginInfo(pluginName: string | null) {
  const [pluginInfo, setPluginInfo] = useState<PluginInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!pluginName) {
      setPluginInfo(null);
      return;
    }

    const loadPluginInfo = async () => {
      setLoading(true);
      setError(null);
      try {
        const info = await getPluginInfo(pluginName);
        setPluginInfo(info);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load plugin info");
      } finally {
        setLoading(false);
      }
    };

    loadPluginInfo();
  }, [pluginName]);

  return {
    pluginInfo,
    loading,
    error,
  };
}

export function usePluginExecution<TInput = any, TOutput = any>() {
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<TOutput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(
    async (pluginName: string, functionName: string, input: TInput) => {
      setExecuting(true);
      setError(null);
      setResult(null);
      try {
        const output = await executePlugin<TInput, TOutput>(
          pluginName,
          functionName,
          input
        );
        setResult(output);
        return output;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to execute plugin";
        setError(errorMsg);
        throw err;
      } finally {
        setExecuting(false);
      }
    },
    []
  );

  return {
    execute,
    executing,
    result,
    error,
  };
}

export function usePluginInstallation() {
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const install = useCallback(async (path: string) => {
    setInstalling(true);
    setError(null);
    try {
      const message = await installPlugin(path);
      return message;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to install plugin";
      setError(errorMsg);
      throw err;
    } finally {
      setInstalling(false);
    }
  }, []);

  const installFromUrl = useCallback(async (url: string) => {
    setInstalling(true);
    setError(null);
    try {
      const message = await installPluginFromUrl(url);
      return message;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to install plugin from URL";
      setError(errorMsg);
      throw err;
    } finally {
      setInstalling(false);
    }
  }, []);

  const discover = useCallback(async () => {
    setInstalling(true);
    setError(null);
    try {
      const count = await discoverPlugins();
      return count;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to discover plugins";
      setError(errorMsg);
      throw err;
    } finally {
      setInstalling(false);
    }
  }, []);

  return {
    install,
    installFromUrl,
    discover,
    installing,
    error,
  };
}
