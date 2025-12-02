use rusqlite::{Connection, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub mod schema;
pub mod migrations;
pub mod operations;

/// Database wrapper with thread-safe connection
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    /// Create a new database connection
    pub fn new(db_path: PathBuf) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        
        Ok(Database {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
    
    /// Get access to the connection
    pub fn with_connection<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Connection) -> Result<R>,
    {
        let conn = self.conn.lock().unwrap();
        f(&*conn)
    }
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Database {
            conn: Arc::clone(&self.conn),
        }
    }
}
