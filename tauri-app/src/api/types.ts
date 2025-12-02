/**
 * Type definitions for Tauri API calls
 */

export interface User {
  id: string;
  name: string;
  email: string;
  emailVerified: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface Session {
  id: string;
  userId: string;
  expiresAt: string;
}

export interface AuthResult {
  user: User;
  sessionId: string;
}

export interface PluginResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface SignUpInput {
  name: string;
  email: string;
  password: string;
}

export interface SignInInput {
  email: string;
  password: string;
}

export interface VerifyEmailInput {
  token: string;
}

export interface RequestPasswordResetInput {
  email: string;
}

export interface ResetPasswordInput {
  token: string;
  newPassword: string;
}
