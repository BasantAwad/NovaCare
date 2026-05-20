"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import { useRouter } from "next/navigation";
import {
  type AuthUser,
  type AuthProfile,
  login as apiLogin,
  logout as apiLogout,
  getMe,
  getAccessToken,
  clearTokens,
} from "@/lib/auth-api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type UserRole = "rover" | "caregiver" | "doctor";

interface AuthState {
  user: AuthUser | null;
  roles: UserRole[];
  activeRole: UserRole | null;
  profile: AuthProfile;
  isAuthenticated: boolean;
  isLoading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string; roles?: string[] }>;
  logout: () => Promise<void>;
  switchRole: (role: UserRole) => void;
  refreshUser: () => Promise<void>;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [state, setState] = useState<AuthState>({
    user: null,
    roles: [],
    activeRole: null,
    profile: {},
    isAuthenticated: false,
    isLoading: true,
  });

  // Restore session on mount
  useEffect(() => {
    const token = getAccessToken();
    if (token) {
      loadUser();
    } else {
      setState((prev) => ({ ...prev, isLoading: false }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadUser = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: true }));
    const res = await getMe();

    if (res.status === "success" && res.data) {
      const roles = res.data.roles as UserRole[];
      // Restore persisted active role or default to first
      const savedRole = typeof window !== "undefined"
        ? (localStorage.getItem("novacare_active_role") as UserRole | null)
        : null;
      const activeRole = savedRole && roles.includes(savedRole) ? savedRole : roles[0] || null;

      setState({
        user: res.data.user,
        roles,
        activeRole,
        profile: res.data.profile || {},
        isAuthenticated: true,
        isLoading: false,
      });
    } else {
      clearTokens();
      setState({
        user: null,
        roles: [],
        activeRole: null,
        profile: {},
        isAuthenticated: false,
        isLoading: false,
      });
    }
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const res = await apiLogin(email, password);

    if (res.status === "success" && res.data) {
      const roles = res.data.roles as UserRole[];
      const activeRole = roles[0] || null;

      if (activeRole) {
        localStorage.setItem("novacare_active_role", activeRole);
      }

      setState({
        user: res.data.user,
        roles,
        activeRole,
        profile: res.data.profile || {},
        isAuthenticated: true,
        isLoading: false,
      });

      return { success: true, roles: res.data.roles };
    }

    return { success: false, error: res.error || "Login failed" };
  }, []);

  const logout = useCallback(async () => {
    await apiLogout();
    localStorage.removeItem("novacare_active_role");
    setState({
      user: null,
      roles: [],
      activeRole: null,
      profile: {},
      isAuthenticated: false,
      isLoading: false,
    });
    router.push("/auth/login");
  }, [router]);

  const switchRole = useCallback((role: UserRole) => {
    if (state.roles.includes(role)) {
      localStorage.setItem("novacare_active_role", role);
      setState((prev) => ({ ...prev, activeRole: role }));
    }
  }, [state.roles]);

  const refreshUser = useCallback(async () => {
    await loadUser();
  }, [loadUser]);

  return (
    <AuthContext.Provider
      value={{
        ...state,
        login,
        logout,
        switchRole,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
