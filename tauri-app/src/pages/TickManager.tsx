import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

interface TickManagerStatus {
  is_running: boolean;
  current_tick: number;
  tick_rate: number;
  active_sessions: number;
  total_clients: number;
}

interface TickEvent {
  tick: number;
  timestamp: number;
  delta_time: number;
}

export default function TickManager() {
  const [status, setStatus] = useState<TickManagerStatus | null>(null);
  const [lastTick, setLastTick] = useState<TickEvent | null>(null);
  const [tickRate, setTickRate] = useState('60');
  const [sessionId, setSessionId] = useState('test-session');
  const [clientId, setClientId] = useState('client-1');
  const [activeSessions, setActiveSessions] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [ticksReceived, setTicksReceived] = useState(0);

  // Fetch status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const result = await invoke<TickManagerStatus>('tick_get_status');
        setStatus(result);
        setError(null);
      } catch (err) {
        setError(String(err));
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  // Listen for tick events
  useEffect(() => {
    const unlisten = listen<TickEvent>('tick', (event) => {
      setLastTick(event.payload);
      setTicksReceived(prev => prev + 1);
    });

    return () => {
      unlisten.then(fn => fn());
    };
  }, []);

  const handleStart = async () => {
    try {
      await invoke('tick_start');
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  };

  const handleStop = async () => {
    try {
      await invoke('tick_stop');
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  };

  const handleSetRate = async () => {
    try {
      const rate = parseInt(tickRate);
      if (rate > 0) {
        await invoke('tick_set_rate', { rate });
        setError(null);
      }
    } catch (err) {
      setError(String(err));
    }
  };

  const handleRegisterSession = async () => {
    try {
      await invoke('tick_register_session', { sessionId });
      await fetchActiveSessions();
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  };

  const handleUnregisterSession = async () => {
    try {
      await invoke('tick_unregister_session', { sessionId });
      await fetchActiveSessions();
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  };

  const handleAddClient = async () => {
    try {
      await invoke('tick_add_client', { sessionId, clientId });
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  };

  const handleRemoveClient = async () => {
    try {
      await invoke('tick_remove_client', { sessionId, clientId });
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  };

  const fetchActiveSessions = async () => {
    try {
      const sessions = await invoke<string[]>('tick_get_active_sessions');
      setActiveSessions(sessions);
    } catch (err) {
      setError(String(err));
    }
  };

  useEffect(() => {
    fetchActiveSessions();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Tick Manager</h1>

        {error && (
          <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Status Card */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Status</h2>
            {status && (
              <dl className="space-y-2">
                <div>
                  <dt className="text-sm font-medium text-gray-500">Running</dt>
                  <dd className="text-base text-gray-900">
                    {status.is_running ? (
                      <span className="text-green-600 font-medium">Yes</span>
                    ) : (
                      <span className="text-red-600 font-medium">No</span>
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Current Tick</dt>
                  <dd className="text-base text-gray-900 font-mono">{status.current_tick}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Tick Rate</dt>
                  <dd className="text-base text-gray-900">{status.tick_rate} TPS</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Active Sessions</dt>
                  <dd className="text-base text-gray-900">{status.active_sessions}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Total Clients</dt>
                  <dd className="text-base text-gray-900">{status.total_clients}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Ticks Received</dt>
                  <dd className="text-base text-gray-900 font-mono">{ticksReceived}</dd>
                </div>
              </dl>
            )}
          </div>

          {/* Last Tick Event */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Last Tick Event</h2>
            {lastTick ? (
              <dl className="space-y-2">
                <div>
                  <dt className="text-sm font-medium text-gray-500">Tick</dt>
                  <dd className="text-base text-gray-900 font-mono">{lastTick.tick}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Timestamp</dt>
                  <dd className="text-base text-gray-900 font-mono">{lastTick.timestamp}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Delta Time</dt>
                  <dd className="text-base text-gray-900">{lastTick.delta_time} ms</dd>
                </div>
              </dl>
            ) : (
              <p className="text-gray-500">No ticks received yet</p>
            )}
          </div>

          {/* Controls */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Controls</h2>
            <div className="space-y-4">
              <div className="flex gap-2">
                <button
                  onClick={handleStart}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
                >
                  Start
                </button>
                <button
                  onClick={handleStop}
                  className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
                >
                  Stop
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Tick Rate (TPS)
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    value={tickRate}
                    onChange={(e) => setTickRate(e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="60"
                  />
                  <button
                    onClick={handleSetRate}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                  >
                    Set Rate
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Session Management */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Session Management</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Session ID
                </label>
                <input
                  type="text"
                  value={sessionId}
                  onChange={(e) => setSessionId(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="test-session"
                />
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleRegisterSession}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                >
                  Register
                </button>
                <button
                  onClick={handleUnregisterSession}
                  className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
                >
                  Unregister
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Client ID
                </label>
                <input
                  type="text"
                  value={clientId}
                  onChange={(e) => setClientId(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="client-1"
                />
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleAddClient}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
                >
                  Add Client
                </button>
                <button
                  onClick={handleRemoveClient}
                  className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 transition"
                >
                  Remove Client
                </button>
              </div>
            </div>
          </div>

          {/* Active Sessions */}
          <div className="bg-white p-6 rounded-lg shadow md:col-span-2">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Active Sessions</h2>
              <button
                onClick={fetchActiveSessions}
                className="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300 transition"
              >
                Refresh
              </button>
            </div>
            {activeSessions.length > 0 ? (
              <ul className="space-y-2">
                {activeSessions.map((session) => (
                  <li
                    key={session}
                    className="p-3 bg-gray-50 rounded border border-gray-200 font-mono text-sm"
                  >
                    {session}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500">No active sessions</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
