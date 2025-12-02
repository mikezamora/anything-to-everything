# Authentication Plugin with Database Host Functions

This plugin demonstrates how to create an authentication system using Extism PDK with database host functions.

## Building

```bash
cargo build --target wasm32-unknown-unknown --release
```

The compiled plugin will be at: `target/wasm32-unknown-unknown/release/auth_plugin.wasm`

## Functions

### `signup`
Create a new user account.

**Input:**
```json
{
  "name": "string",
  "email": "string", 
  "password": "string"
}
```

**Output:**
```json
{
  "success": true,
  "user_uuid": "string",
  "message": "User created successfully"
}
```

### `login`
Authenticate a user and create a session.

**Input:**
```json
{
  "email": "string",
  "password": "string"
}
```

**Output:**
```json
{
  "success": true,
  "session_id": "string",
  "user": {
    "uuid": "string",
    "name": "string",
    "email": "string"
  }
}
```

### `verify_session`
Check if a session is valid.

**Input:**
```json
{
  "session_id": "string"
}
```

**Output:**
```json
{
  "success": true,
  "valid": true,
  "user_uuid": "string"
}
```

### `logout`
End a user session.

**Input:**
```json
{
  "session_id": "string"
}
```

**Output:**
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

## Host Functions Used

This plugin requires the following host functions to be provided by the Tauri app:

- `db_create_user(json) -> json` - Create user in database
- `db_get_user_by_email(email) -> json` - Find user by email
- `db_get_user_by_uuid(uuid) -> json` - Find user by UUID
- `db_update_user_password(json) -> json` - Update user password
- `db_create_session(json) -> json` - Create session
- `db_get_session(session_id) -> json` - Get session details
- `db_delete_session(session_id) -> json` - Delete session

## Testing

Once the Tauri host functions are implemented, you can test with:

```bash
# Via Tauri commands
tauri invoke execute_plugin --args '{"plugin": "auth", "function": "signup", "input": {...}}'
```
