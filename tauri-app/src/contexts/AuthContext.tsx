import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import * as authApi from '../api/auth';
import type { User, AuthResult } from '../api/types';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const SESSION_ID_KEY = 'session_id';
const USER_ID_KEY = 'user_id';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    async function checkSession() {
      try {
        const sessionId = localStorage.getItem(SESSION_ID_KEY);
        const userId = localStorage.getItem(USER_ID_KEY);

        if (sessionId && userId) {
          // Verify session is still valid
          await authApi.verifySession(sessionId);
          
          // Fetch user data
          const userData = await authApi.getCurrentUser(userId);
          setUser(userData);
        }
      } catch (error) {
        // Session invalid or expired
        console.error('Session verification failed:', error);
        localStorage.removeItem(SESSION_ID_KEY);
        localStorage.removeItem(USER_ID_KEY);
      } finally {
        setLoading(false);
      }
    }

    checkSession();
  }, []);

  async function handleSignIn(email: string, password: string) {
    const result: AuthResult = await authApi.signIn({ email, password });
    
    // Store session info
    localStorage.setItem(SESSION_ID_KEY, result.sessionId);
    localStorage.setItem(USER_ID_KEY, result.user.id);
    
    setUser(result.user);
  }

  async function handleSignUp(name: string, email: string, password: string) {
    const result: AuthResult = await authApi.signUp({ name, email, password });
    
    // Store session info
    localStorage.setItem(SESSION_ID_KEY, result.sessionId);
    localStorage.setItem(USER_ID_KEY, result.user.id);
    
    setUser(result.user);
  }

  async function handleSignOut() {
    try {
      await authApi.signOut();
    } finally {
      // Clear local state even if API call fails
      localStorage.removeItem(SESSION_ID_KEY);
      localStorage.removeItem(USER_ID_KEY);
      setUser(null);
    }
  }

  const value: AuthContextType = {
    user,
    loading,
    signIn: handleSignIn,
    signUp: handleSignUp,
    signOut: handleSignOut,
    isAuthenticated: !!user,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
