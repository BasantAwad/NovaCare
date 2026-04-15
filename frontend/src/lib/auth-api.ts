/**
 * NovaCare — Frontend Auth API Client
 *
 * Communicates with the Flask auth-backend at the URL defined
 * in NEXT_PUBLIC_AUTH_API_URL (defaults to http://localhost:5001).
 *
 * All tokens are stored in localStorage and attached as Bearer headers.
 */

const AUTH_API = process.env.NEXT_PUBLIC_AUTH_API_URL || "http://localhost:5001";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AuthUser {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  profile_picture_url?: string | null;
  is_email_verified?: boolean;
  created_at?: string;
  last_login_at?: string | null;
}

export interface AuthProfile {
  rover?: Record<string, unknown>;
  caregiver?: Record<string, unknown>;
  doctor?: Record<string, unknown>;
  caregiver_verification?: string;
  doctor_verification?: string;
  specialization?: string;
}

export interface ReferenceData {
  countries: Array<{ id: string; name: string; iso_code?: string }>;
  specializations: Array<{ id: string; name: string; code?: string }>;
  medication_catalog: Array<{ id: string; name: string }>;
  // Legacy fields that the signup form may still reference
  health_conditions?: Array<{ id: string; name: string }>;
  medications?: Array<{ id: string; name: string }>;
  allergies?: Array<{ id: string; name: string }>;
  id_types?: Array<{ id: string; name: string }>;
  relationship_types?: Array<{ id: string; relationship: string }>;
  clinic_organizations?: Array<{ id: string; name: string }>;
}

interface ApiResponse<T = unknown> {
  status: "success" | "error";
  data?: T;
  error?: string;
  message?: string;
}

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

const TOKEN_KEY = "novacare_access_token";
const REFRESH_KEY = "novacare_refresh_token";

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(REFRESH_KEY);
}

function saveTokens(access: string, refresh: string) {
  localStorage.setItem(TOKEN_KEY, access);
  localStorage.setItem(REFRESH_KEY, refresh);
}

export function clearTokens() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_KEY);
}

// ---------------------------------------------------------------------------
// Fetch wrapper
// ---------------------------------------------------------------------------

async function authFetch<T = unknown>(
  path: string,
  options: RequestInit = {},
): Promise<ApiResponse<T>> {
  const url = `${AUTH_API}${path}`;
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
    const json = await res.json();
    return json as ApiResponse<T>;
  } catch {
    return { status: "error", error: "Could not connect to authentication service." };
  }
}

// ---------------------------------------------------------------------------
// Auth API Functions
// ---------------------------------------------------------------------------

/** Login with email + password */
export async function login(
  email: string,
  password: string,
): Promise<ApiResponse<{ user: AuthUser; roles: string[]; profile: AuthProfile; access_token: string; refresh_token: string }>> {
  const res = await authFetch<{ user: AuthUser; roles: string[]; profile: AuthProfile; access_token: string; refresh_token: string }>(
    "/api/auth/login",
    { method: "POST", body: JSON.stringify({ email, password }) },
  );

  if (res.status === "success" && res.data) {
    saveTokens(res.data.access_token, res.data.refresh_token);
  }
  return res;
}

/** Login with Google ID */
export async function loginWithGoogle(
  googleId: string,
  email: string,
): Promise<ApiResponse<{ user: AuthUser; roles: string[]; profile: AuthProfile; access_token: string; refresh_token: string }>> {
  const res = await authFetch<{ user: AuthUser; roles: string[]; profile: AuthProfile; access_token: string; refresh_token: string }>(
    "/api/auth/login/google",
    { method: "POST", body: JSON.stringify({ google_id: googleId, email }) },
  );

  if (res.status === "success" && res.data) {
    saveTokens(res.data.access_token, res.data.refresh_token);
  }
  return res;
}

/** Get current user info */
export async function getMe(): Promise<ApiResponse<{ user: AuthUser; roles: string[]; profile: AuthProfile }>> {
  return authFetch("/api/auth/me");
}

/** Logout (revoke session) */
export async function logout(): Promise<ApiResponse> {
  const res = await authFetch("/api/auth/logout", { method: "POST" });
  clearTokens();
  return res;
}

/** Refresh access token */
export async function refreshAccessToken(): Promise<ApiResponse<{ access_token: string }>> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return { status: "error", error: "No refresh token" };

  const res = await authFetch<{ access_token: string }>("/api/auth/refresh", {
    method: "POST",
    body: JSON.stringify({ refresh_token: refreshToken }),
  });

  if (res.status === "success" && res.data) {
    localStorage.setItem(TOKEN_KEY, res.data.access_token);
  }
  return res;
}

// ---------------------------------------------------------------------------
// Signup
// ---------------------------------------------------------------------------

export async function signupRover(data: Record<string, unknown>): Promise<ApiResponse> {
  const res = await authFetch("/api/auth/signup/rover", {
    method: "POST",
    body: JSON.stringify(data),
  });
  if (res.status === "success" && (res.data as Record<string, unknown>)?.access_token) {
    const d = res.data as Record<string, string>;
    saveTokens(d.access_token, d.refresh_token);
  }
  return res;
}

export async function signupCaregiver(data: Record<string, unknown>): Promise<ApiResponse> {
  const res = await authFetch("/api/auth/signup/caregiver", {
    method: "POST",
    body: JSON.stringify(data),
  });
  if (res.status === "success" && (res.data as Record<string, unknown>)?.access_token) {
    const d = res.data as Record<string, string>;
    saveTokens(d.access_token, d.refresh_token);
  }
  return res;
}

export async function signupDoctor(data: Record<string, unknown>): Promise<ApiResponse> {
  const res = await authFetch("/api/auth/signup/doctor", {
    method: "POST",
    body: JSON.stringify(data),
  });
  if (res.status === "success" && (res.data as Record<string, unknown>)?.access_token) {
    const d = res.data as Record<string, string>;
    saveTokens(d.access_token, d.refresh_token);
  }
  return res;
}

// ---------------------------------------------------------------------------
// Password & Verification
// ---------------------------------------------------------------------------

export async function forgotPassword(email: string): Promise<ApiResponse> {
  return authFetch("/api/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export async function resetPassword(token: string, newPassword: string): Promise<ApiResponse> {
  return authFetch("/api/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ token, new_password: newPassword }),
  });
}

export async function verifyEmail(token: string): Promise<ApiResponse> {
  return authFetch("/api/auth/verify-email", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

// ---------------------------------------------------------------------------
// Reference Data
// ---------------------------------------------------------------------------

export async function getReferenceData(): Promise<ApiResponse<ReferenceData>> {
  return authFetch<ReferenceData>("/api/auth/reference-data");
}
