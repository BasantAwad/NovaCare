/**
 * NovaCare — Authentication API Client
 *
 * Handles identity management, session persistence, and RBAC communication 
 * with the Flask auth-backend service.
 */

import { getDynamicUrl } from "./utils";

const AUTH_API = process.env.NEXT_PUBLIC_AUTH_API_URL || "http://localhost:5001";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AuthUser {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  profile_picture_url?: string;
  is_email_verified: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuthProfile {
  rover?: Record<string, any>;
  caregiver?: Record<string, any>;
  doctor?: Record<string, any>;
}

export interface ApiResponse<T = any> {
  status: "success" | "error";
  data?: T;
  error?: string;
}

export interface AuthResponseData {
  user: AuthUser;
  roles: string[];
  profile: AuthProfile;
  access_token: string;
  refresh_token: string;
}

export type ReferenceData = Record<string, any[]>;

// ---------------------------------------------------------------------------
// Token Management
// ---------------------------------------------------------------------------

const ACCESS_TOKEN_KEY = "novacare_access_token";
const REFRESH_TOKEN_KEY = "novacare_refresh_token";

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function setTokens(access: string, refresh: string) {
  if (typeof window === "undefined") return;
  localStorage.setItem(ACCESS_TOKEN_KEY, access);
  localStorage.setItem(REFRESH_TOKEN_KEY, refresh);
}

export function clearTokens() {
  if (typeof window === "undefined") return;
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

// ---------------------------------------------------------------------------
// Fetch Wrapper
// ---------------------------------------------------------------------------

async function authFetch<T = any>(
  path: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  const url = `${getDynamicUrl(AUTH_API)}${path}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  const token = getAccessToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    const res = await fetch(url, { ...options, headers });
    
    // Handle unauthorized (expired token)
    if (res.status === 401 && getRefreshToken()) {
      // Logic for token refresh could go here if needed
      // For now, we'll just let the caller handle errors
    }

    const json = await res.json();
    return json as ApiResponse<T>;
  } catch (err) {
    return { status: "error", error: "Could not connect to authentication service." };
  }
}

// ---------------------------------------------------------------------------
// Auth API Functions
// ---------------------------------------------------------------------------

/** Authenticate user with email + password */
export async function login(email: string, password: string): Promise<ApiResponse<AuthResponseData>> {
  const res = await authFetch<AuthResponseData>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });

  if (res.status === "success" && res.data) {
    setTokens(res.data.access_token, res.data.refresh_token);
  }

  return res;
}

/** Authenticate user with Google ID */
export async function loginWithGoogle(google_id: string, email: string): Promise<ApiResponse<AuthResponseData>> {
  const res = await authFetch<AuthResponseData>("/api/auth/login/google", {
    method: "POST",
    body: JSON.stringify({ google_id, email }),
  });

  if (res.status === "success" && res.data) {
    setTokens(res.data.access_token, res.data.refresh_token);
  }

  return res;
}

/** Signup a new Rover */
export async function signupRover(data: Record<string, any>): Promise<ApiResponse<AuthResponseData>> {
  const res = await authFetch<AuthResponseData>("/api/auth/signup/rover", {
    method: "POST",
    body: JSON.stringify(data),
  });

  if (res.status === "success" && res.data) {
    setTokens(res.data.access_token, res.data.refresh_token);
  }

  return res;
}

/** Signup a new Caregiver */
export async function signupCaregiver(data: Record<string, any>): Promise<ApiResponse<AuthResponseData>> {
  const res = await authFetch<AuthResponseData>("/api/auth/signup/caregiver", {
    method: "POST",
    body: JSON.stringify(data),
  });

  if (res.status === "success" && res.data) {
    setTokens(res.data.access_token, res.data.refresh_token);
  }

  return res;
}

/** Signup a new Doctor */
export async function signupDoctor(data: Record<string, any>): Promise<ApiResponse<AuthResponseData>> {
  const res = await authFetch<AuthResponseData>("/api/auth/signup/doctor", {
    method: "POST",
    body: JSON.stringify(data),
  });

  if (res.status === "success" && res.data) {
    setTokens(res.data.access_token, res.data.refresh_token);
  }

  return res;
}

/** Fetch current user info and profile */
export async function getMe(): Promise<ApiResponse<{ user: AuthUser; roles: string[]; profile: AuthProfile }>> {
  return authFetch("/api/auth/me");
}

/** Revoke current session */
export async function logout(): Promise<ApiResponse<{ message: string }>> {
  const res = await authFetch<{ message: string }>("/api/auth/logout", {
    method: "POST",
  });
  clearTokens();
  return res;
}

/** Request password reset link */
export async function forgotPassword(email: string): Promise<ApiResponse<{ message: string }>> {
  return authFetch("/api/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

/** Verify email with token */
export async function verifyEmail(token: string): Promise<ApiResponse<{ message: string }>> {
  return authFetch("/api/auth/verify-email", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

/** Fetch reference data for signup forms */
export async function getReferenceData(): Promise<ApiResponse<Record<string, any[]>>> {
  return authFetch("/api/auth/reference-data");
}
