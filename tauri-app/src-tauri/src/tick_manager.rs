use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time;
use tauri::{AppHandle, Emitter};

/// Tick event data sent to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickEvent {
    pub tick: u64,
    pub timestamp: u64,
    pub delta_time: u64,
}

/// Session-specific tick event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTickEvent {
    pub session_id: String,
    pub tick: u64,
    pub timestamp: u64,
    pub delta_time: u64,
    pub client_count: usize,
}

/// Session information
#[derive(Debug, Clone)]
struct SessionInfo {
    last_tick: u64,
    clients: HashSet<String>,
}

/// Tick manager status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickManagerStatus {
    pub is_running: bool,
    pub current_tick: u64,
    pub tick_rate: u32,
    pub active_sessions: usize,
    pub total_clients: usize,
}

/// Server-side authoritative tick manager
/// Ensures all clients stay synchronized with a fixed tick rate
pub struct TickManager {
    tick_rate: u32,
    current_tick: u64,
    last_tick_time: u64,
    is_running: bool,
    sessions: HashMap<String, SessionInfo>,
}

impl TickManager {
    pub fn new(tick_rate: u32) -> Self {
        Self {
            tick_rate,
            current_tick: 0,
            last_tick_time: 0,
            is_running: false,
            sessions: HashMap::new(),
        }
    }

    pub fn start(&mut self) -> Result<(), String> {
        if self.is_running {
            return Err("Tick manager is already running".to_string());
        }

        self.is_running = true;
        self.last_tick_time = current_timestamp();
        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), String> {
        if !self.is_running {
            return Err("Tick manager is not running".to_string());
        }

        self.is_running = false;
        Ok(())
    }

    pub fn advance_tick(&mut self) -> TickEvent {
        let now = current_timestamp();
        let delta_time = if self.last_tick_time > 0 {
            now - self.last_tick_time
        } else {
            0
        };

        self.current_tick += 1;
        self.last_tick_time = now;

        // Update session tracking
        for session in self.sessions.values_mut() {
            session.last_tick = self.current_tick;
        }

        TickEvent {
            tick: self.current_tick,
            timestamp: now,
            delta_time,
        }
    }

    pub fn get_current_tick(&self) -> u64 {
        self.current_tick
    }

    pub fn get_current_tick_time(&self) -> u64 {
        self.last_tick_time
    }

    pub fn register_session(&mut self, session_id: String) {
        if !self.sessions.contains_key(&session_id) {
            self.sessions.insert(
                session_id.clone(),
                SessionInfo {
                    last_tick: self.current_tick,
                    clients: HashSet::new(),
                },
            );
            tracing::debug!("Registered session: {}", session_id);
        }
    }

    pub fn unregister_session(&mut self, session_id: &str) {
        if self.sessions.remove(session_id).is_some() {
            tracing::debug!("Unregistered session: {}", session_id);
        }
    }

    pub fn add_client_to_session(&mut self, session_id: String, client_id: String) {
        if let Some(session) = self.sessions.get_mut(&session_id) {
            session.clients.insert(client_id.clone());
            tracing::debug!("Added client {} to session {}", client_id, session_id);
        }
    }

    pub fn remove_client_from_session(&mut self, session_id: &str, client_id: &str) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.clients.remove(client_id);
            tracing::debug!("Removed client {} from session {}", client_id, session_id);

            // Clean up empty sessions
            if session.clients.is_empty() {
                self.unregister_session(session_id);
            }
        }
    }

    pub fn get_session_info(&self, session_id: &str) -> Option<(u64, usize)> {
        self.sessions.get(session_id).map(|session| {
            (session.last_tick, session.clients.len())
        })
    }

    pub fn get_active_sessions(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }

    pub fn get_tick_difference(&self, _session_id: &str, client_tick: u64) -> i64 {
        self.current_tick as i64 - client_tick as i64
    }

    pub fn is_session_behind(&self, session_id: &str, client_tick: u64, threshold: i64) -> bool {
        self.get_tick_difference(session_id, client_tick) > threshold
    }

    pub fn get_tick_rate(&self) -> u32 {
        self.tick_rate
    }

    pub fn set_tick_rate(&mut self, new_rate: u32) -> Result<(), String> {
        if new_rate == 0 {
            return Err("Tick rate must be greater than 0".to_string());
        }

        self.tick_rate = new_rate;
        tracing::info!("Tick rate changed to {} ticks/second", new_rate);
        Ok(())
    }

    pub fn get_status(&self) -> TickManagerStatus {
        let total_clients: usize = self.sessions.values().map(|s| s.clients.len()).sum();

        TickManagerStatus {
            is_running: self.is_running,
            current_tick: self.current_tick,
            tick_rate: self.tick_rate,
            active_sessions: self.sessions.len(),
            total_clients,
        }
    }

    pub fn is_running(&self) -> bool {
        self.is_running
    }

    pub fn get_session_tick_events(&self) -> Vec<SessionTickEvent> {
        let now = current_timestamp();
        let delta_time = if self.last_tick_time > 0 {
            now - self.last_tick_time
        } else {
            0
        };

        self.sessions
            .iter()
            .map(|(session_id, session)| SessionTickEvent {
                session_id: session_id.clone(),
                tick: self.current_tick,
                timestamp: now,
                delta_time,
                client_count: session.clients.len(),
            })
            .collect()
    }
}

/// Get current Unix timestamp in milliseconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Start the tick loop in a background task
pub async fn start_tick_loop(
    tick_manager: Arc<RwLock<TickManager>>,
    app_handle: AppHandle,
) {
    // Get tick rate from manager
    let tick_rate = {
        let manager = tick_manager.read().await;
        manager.get_tick_rate()
    };

    let interval_ms = 1000 / tick_rate as u64;
    let mut interval = time::interval(Duration::from_millis(interval_ms));

    loop {
        interval.tick().await;

        // Check if still running
        let is_running = {
            let manager = tick_manager.read().await;
            manager.is_running()
        };

        if !is_running {
            break;
        }

        // Advance tick
        let (tick_event, session_events) = {
            let mut manager = tick_manager.write().await;
            let tick_event = manager.advance_tick();
            let session_events = manager.get_session_tick_events();
            (tick_event, session_events)
        };

        // Emit global tick event
        let _ = app_handle.emit("tick", &tick_event);

        // Emit session-specific tick events
        for session_event in session_events {
            let event_name = format!("tick:{}", session_event.session_id);
            let _ = app_handle.emit(&event_name, &session_event);
        }
    }

    tracing::info!("Tick loop stopped");
}
