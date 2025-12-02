import type {
  AuthResult,
  PluginResult,
  SignUpInput,
  SignInInput,
  VerifyEmailInput,
  RequestPasswordResetInput,
  ResetPasswordInput,
  User,
  Session,
} from './types';
import { executePlugin } from './plugins';

/**
 * Execute an auth plugin function via Tauri command
 */
async function executeAuthPlugin<T>(
  functionName: string,
  input: unknown
): Promise<T> {
  const result = await executePlugin<unknown, PluginResult<T>>(
    'auth-plugin',
    functionName,
    input
  );

  if (!result.success || !result.data) {
    throw new Error(result.error || 'Authentication failed');
  }

  return result.data;
}

/**
 * Sign up a new user
 */
export async function signUp(data: SignUpInput): Promise<AuthResult> {
  return executeAuthPlugin<AuthResult>('signup', data);
}

/**
 * Sign in an existing user
 */
export async function signIn(data: SignInInput): Promise<AuthResult> {
  return executeAuthPlugin<AuthResult>('login', data);
}

/**
 * Sign out the current user
 */
export async function signOut(): Promise<void> {
  return executeAuthPlugin<void>('logout', {});
}

/**
 * Verify the current session
 */
export async function verifySession(sessionId: string): Promise<Session> {
  return executeAuthPlugin<Session>('verify_session', { sessionId });
}

/**
 * Get the current user
 */
export async function getCurrentUser(userId: string): Promise<User> {
  return executeAuthPlugin<User>('get_user', { userId });
}

/**
 * Verify email address with token
 */
export async function verifyEmail(data: VerifyEmailInput): Promise<void> {
  return executeAuthPlugin<void>('verify_email', data);
}

/**
 * Request password reset email
 */
export async function requestPasswordReset(data: RequestPasswordResetInput): Promise<void> {
  return executeAuthPlugin<void>('request_password_reset', data);
}

/**
 * Reset password with token
 */
export async function resetPassword(data: ResetPasswordInput): Promise<void> {
  return executeAuthPlugin<void>('reset_password', data);
}
