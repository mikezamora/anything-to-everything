import { useState } from "react";
import reactLogo from "./assets/react.svg";
import "./App.css";
import { usePlugins, usePluginExecution, usePluginInstallation, usePluginInfo } from "./hooks/usePlugins";

function App() {
  const { plugins, loading, error, loadPlugins } = usePlugins();
  const { execute, executing, result, error: execError } = usePluginExecution();
  const { installFromUrl, installing } = usePluginInstallation();
  const [selectedPlugin, setSelectedPlugin] = useState<string>("");
  const [selectedFunction, setSelectedFunction] = useState<string>("");
  const [inputData, setInputData] = useState<string>('"Hello, World!"');
  const [pluginUrl, setPluginUrl] = useState<string>("");
  
  const { pluginInfo } = usePluginInfo(selectedPlugin);

  async function handleExecute() {
    if (!selectedPlugin || !selectedFunction) return;
    try {
      const input = JSON.parse(inputData);
      await execute(selectedPlugin, selectedFunction, input);
    } catch (err) {
      console.error("Failed to execute plugin:", err);
    }
  }
  
  // Update function when plugin changes
  const handlePluginChange = (pluginName: string) => {
    setSelectedPlugin(pluginName);
    setSelectedFunction("");
  };

  async function handleInstallFromUrl() {
    if (!pluginUrl) return;
    try {
      await installFromUrl(pluginUrl);
      setPluginUrl("");
      await loadPlugins();
    } catch (err) {
      console.error("Failed to install plugin:", err);
    }
  }

  return (
    <main className="container">
      <h1>WASM Plugin System</h1>

      <div className="row">
        <a href="https://vite.dev" target="_blank">
          <img src="/vite.svg" className="logo vite" alt="Vite logo" />
        </a>
        <a href="https://tauri.app" target="_blank">
          <img src="/tauri.svg" className="logo tauri" alt="Tauri logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>

      <div style={{ marginTop: "2rem" }}>
        <h2>Install Plugin from URL</h2>
        <form
          className="row"
          onSubmit={(e) => {
            e.preventDefault();
            handleInstallFromUrl();
          }}
        >
          <input
            type="url"
            value={pluginUrl}
            onChange={(e) => setPluginUrl(e.target.value)}
            placeholder="https://example.com/plugin.wasm or https://example.com/plugin.json"
            style={{ flex: 1 }}
          />
          <button type="submit" disabled={installing || !pluginUrl}>
            {installing ? "Installing..." : "Install"}
          </button>
        </form>
        <p style={{ fontSize: "0.9rem", marginTop: "0.5rem", opacity: 0.7 }}>
          Enter a direct link to a .wasm file or a plugin.json manifest
        </p>
      </div>

      <div style={{ marginTop: "2rem" }}>
        <h2>Available Plugins</h2>
        {loading && <p>Loading plugins...</p>}
        {error && <p style={{ color: "red" }}>Error: {error}</p>}
        {!loading && !error && plugins.length === 0 && (
          <p>No plugins found. Install plugins to get started.</p>
        )}
        {!loading && plugins.length > 0 && (
          <ul>
            {plugins.map((plugin) => (
              <li key={plugin.name}>
                <strong>{plugin.name}</strong> v{plugin.version} - {plugin.description}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div style={{ marginTop: "2rem" }}>
        <h2>Execute Plugin</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleExecute();
          }}
          style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
        >
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <select
              value={selectedPlugin}
              onChange={(e) => handlePluginChange(e.target.value)}
              disabled={plugins.length === 0}
              style={{ flex: 1, padding: "0.5rem" }}
            >
              <option value="">Select a plugin...</option>
              {plugins.map((plugin) => (
                <option key={plugin.name} value={plugin.name}>
                  {plugin.name}
                </option>
              ))}
            </select>
          </div>
          
          {selectedPlugin && pluginInfo && (
            <div>
              <p style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>
                <strong>Description:</strong> {pluginInfo.description}
              </p>
              {pluginInfo.entry_points.length > 0 ? (
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <select
                    value={selectedFunction}
                    onChange={(e) => setSelectedFunction(e.target.value)}
                    style={{ flex: 1, padding: "0.5rem" }}
                  >
                    <option value="">Select a function...</option>
                    {pluginInfo.entry_points.map((ep) => (
                      <option key={ep.name} value={ep.name}>
                        {ep.name} - {ep.description}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div>
                  <label style={{ fontSize: "0.9rem" }}>Function name:</label>
                  <input
                    type="text"
                    value={selectedFunction}
                    onChange={(e) => setSelectedFunction(e.target.value)}
                    placeholder="e.g., count_vowels"
                    style={{ width: "100%", padding: "0.5rem", marginTop: "0.25rem" }}
                  />
                  <p style={{ fontSize: "0.8rem", opacity: 0.7, marginTop: "0.25rem" }}>
                    No entry points defined. Enter the WASM function name manually.
                  </p>
                </div>
              )}
            </div>
          )}
          
          <div>
            <label style={{ fontSize: "0.9rem" }}>Input (JSON):</label>
            <textarea
              value={inputData}
              onChange={(e) => setInputData(e.target.value)}
              placeholder='"Hello World"'
              rows={3}
              style={{ width: "100%", padding: "0.5rem", marginTop: "0.25rem" }}
            />
          </div>
          
          <button type="submit" disabled={executing || !selectedPlugin || !selectedFunction} style={{ padding: "0.5rem" }}>
            {executing ? "Executing..." : "Execute"}
          </button>
        </form>
        {execError && <p style={{ color: "red" }}>Error: {execError}</p>}
        {result && (
          <div style={{ marginTop: "1rem" }}>
            <strong>Result:</strong>
            <pre style={{ textAlign: "left", background: "#1a1a1a", padding: "1rem" }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </main>
  );
}

export default App;
