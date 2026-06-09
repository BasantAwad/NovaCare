/**
 * NovaCare Voice Assistant — Centralized Route Registry
 *
 * Maps EVERY real Next.js page in the app to voice aliases.
 * This is the single source of truth for navigation.
 * NO hardcoded paths anywhere else.
 */

export type UserRole = 'medical' | 'guardian' | 'admin' | 'rover' | 'unknown';

export interface RouteEntry {
  /** Unique key used by the command registry */
  id: string;
  /** The exact Next.js path */
  path: string;
  /** Human-readable label for TTS */
  label: string;
  /** Roles that can access this route */
  roles: UserRole[];
  /** All voice phrases that should resolve to this route */
  aliases: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// REAL APPLICATION ROUTES  (derived directly from the file-system audit)
// ─────────────────────────────────────────────────────────────────────────────
export const APP_ROUTES: RouteEntry[] = [
  // ── Public ──────────────────────────────────────────────────────────────
  {
    id: 'HOME',
    path: '/',
    label: 'home',
    roles: ['medical', 'guardian', 'admin', 'rover', 'unknown'],
    aliases: ['home', 'landing', 'start', 'main page', 'homepage', 'go home'],
  },

  // ── Auth ────────────────────────────────────────────────────────────────
  {
    id: 'AUTH_LOGIN',
    path: '/auth/login',
    label: 'login',
    roles: ['unknown'],
    aliases: ['login', 'sign in', 'log in', 'signin', 'auth login', 'authentication'],
  },
  {
    id: 'AUTH_SIGNUP',
    path: '/auth/signup',
    label: 'sign up',
    roles: ['unknown'],
    aliases: ['signup', 'sign up', 'register', 'create account', 'new account'],
  },
  {
    id: 'AUTH_FORGOT_PASSWORD',
    path: '/auth/forgot-password',
    label: 'forgot password',
    roles: ['unknown'],
    aliases: ['forgot password', 'reset password', 'forgot my password', 'password reset'],
  },
  {
    id: 'AUTH_VERIFY_EMAIL',
    path: '/auth/verify-email',
    label: 'verify email',
    roles: ['unknown'],
    aliases: ['verify email', 'email verification', 'confirm email'],
  },

  // ── Medical Professional ────────────────────────────────────────────────
  {
    id: 'MEDICAL_DASHBOARD',
    path: '/medical',
    label: 'medical dashboard',
    roles: ['medical'],
    aliases: [
      'medical', 'medical dashboard', 'doctor dashboard', 'medical home',
      'dashboard', 'home', 'main', 'overview',
    ],
  },
  {
    id: 'MEDICAL_APPOINTMENTS',
    path: '/medical/appointments',
    label: 'appointments',
    roles: ['medical'],
    aliases: [
      'appointments', 'my appointments', 'schedule', 'calendar',
      'appointment list', 'appointments page', 'open appointments',
    ],
  },
  {
    id: 'MEDICAL_CARE_PLAN',
    path: '/medical/care-plan',
    label: 'care plan',
    roles: ['medical'],
    aliases: [
      'care plan', 'careplan', 'treatment plan', 'patient care plan',
      'plan', 'care', 'patient plan',
    ],
  },
  {
    id: 'MEDICAL_MEDICATIONS',
    path: '/medical/medications',
    label: 'medications',
    roles: ['medical'],
    aliases: [
      'medications', 'meds', 'prescriptions', 'drugs', 'medicine',
      'medication list', 'patient medications',
    ],
  },
  {
    id: 'MEDICAL_RECORDS',
    path: '/medical/records',
    label: 'medical records',
    roles: ['medical'],
    aliases: [
      'records', 'medical records', 'patient records', 'health records',
      'files', 'documents',
    ],
  },
  {
    id: 'MEDICAL_SETTINGS',
    path: '/medical/settings',
    label: 'settings',
    roles: ['medical'],
    aliases: [
      'settings', 'preferences', 'configuration', 'account settings',
      'my settings', 'open settings', 'go to settings', 'settings page',
      'profile settings',
    ],
  },
  {
    id: 'MEDICAL_VITALS',
    path: '/medical/vitals',
    label: 'vitals',
    roles: ['medical'],
    aliases: [
      'vitals', 'vital signs', 'health vitals', 'patient vitals',
      'signs', 'health metrics', 'biometrics',
    ],
  },

  // ── Guardian ────────────────────────────────────────────────────────────
  {
    id: 'GUARDIAN_DASHBOARD',
    path: '/guardian',
    label: 'guardian dashboard',
    roles: ['guardian'],
    aliases: [
      'guardian', 'guardian dashboard', 'guardian home', 'family dashboard',
      'dashboard', 'home', 'main', 'overview',
    ],
  },
  {
    id: 'GUARDIAN_ACTIVITY',
    path: '/guardian/activity',
    label: 'activity',
    roles: ['guardian'],
    aliases: [
      'activity', 'activities', 'patient activity', 'daily activity',
      'activity log', 'activity monitor',
    ],
  },
  {
    id: 'GUARDIAN_COMMUNICATION',
    path: '/guardian/communication',
    label: 'communication',
    roles: ['guardian'],
    aliases: [
      'communication', 'messages', 'chat', 'contact', 'messaging',
      'communicate', 'inbox',
    ],
  },
  {
    id: 'GUARDIAN_MEDICATIONS',
    path: '/guardian/medications',
    label: 'medications',
    roles: ['guardian'],
    aliases: [
      'medications', 'meds', 'prescriptions', 'medicine', 'medication schedule',
    ],
  },
  {
    id: 'GUARDIAN_SETTINGS',
    path: '/guardian/settings',
    label: 'settings',
    roles: ['guardian'],
    aliases: [
      'settings', 'preferences', 'configuration', 'account settings',
      'my settings', 'open settings', 'go to settings', 'settings page',
    ],
  },

  // ── Admin ───────────────────────────────────────────────────────────────
  {
    id: 'ADMIN_DASHBOARD',
    path: '/admin',
    label: 'admin dashboard',
    roles: ['admin'],
    aliases: [
      'admin', 'admin dashboard', 'administration', 'admin home',
      'dashboard', 'home', 'main', 'overview', 'control panel',
    ],
  },
  {
    id: 'ADMIN_LOGS',
    path: '/admin/logs',
    label: 'system logs',
    roles: ['admin'],
    aliases: [
      'logs', 'system logs', 'audit logs', 'log viewer', 'event logs',
      'activity logs',
    ],
  },
  {
    id: 'ADMIN_SERVICES',
    path: '/admin/services',
    label: 'services',
    roles: ['admin'],
    aliases: [
      'services', 'service manager', 'system services', 'manage services',
      'service list',
    ],
  },
  {
    id: 'ADMIN_SETTINGS',
    path: '/admin/settings',
    label: 'admin settings',
    roles: ['admin'],
    aliases: [
      'settings', 'admin settings', 'system settings', 'preferences',
      'configuration', 'open settings', 'go to settings', 'settings page',
    ],
  },
  {
    id: 'ADMIN_USERS',
    path: '/admin/users',
    label: 'users',
    roles: ['admin'],
    aliases: [
      'users', 'user management', 'manage users', 'user list',
      'patients', 'staff', 'accounts',
    ],
  },

  // ── Rover (Patient Touchscreen) ─────────────────────────────────────────
  {
    id: 'ROVER_DASHBOARD',
    path: '/rover',
    label: 'rover dashboard',
    roles: ['rover'],
    aliases: [
      'rover', 'rover dashboard', 'patient interface', 'patient home',
      'dashboard', 'home', 'main', 'overview',
    ],
  },
  {
    id: 'ROVER_EMERGENCY',
    path: '/rover/emergency',
    label: 'emergency',
    roles: ['rover'],
    aliases: [
      'emergency', 'sos', 'help', 'emergency page', 'call for help',
      'emergency alert', 'urgent',
    ],
  },
  {
    id: 'ROVER_ENTERTAINMENT',
    path: '/rover/entertainment',
    label: 'entertainment',
    roles: ['rover'],
    aliases: [
      'entertainment', 'media', 'music', 'videos', 'tv', 'play',
      'relax', 'entertainment page',
    ],
  },
  {
    id: 'ROVER_HEALTH',
    path: '/rover/health',
    label: 'health',
    roles: ['rover'],
    aliases: [
      'health', 'my health', 'health page', 'health status', 'vitals',
      'health info',
    ],
  },
  {
    id: 'ROVER_HELP',
    path: '/rover/help',
    label: 'help',
    roles: ['rover'],
    aliases: [
      'help', 'assistance', 'support', 'help page', 'how to',
      'instructions', 'tutorial',
    ],
  },
  {
    id: 'ROVER_MEDICATIONS',
    path: '/rover/medications',
    label: 'medications',
    roles: ['rover'],
    aliases: [
      'medications', 'meds', 'my medications', 'medicine', 'pills',
      'medication reminder',
    ],
  },
  {
    id: 'ROVER_NAVIGATE',
    path: '/rover/navigate',
    label: 'navigation',
    roles: ['rover'],
    aliases: [
      'navigate', 'navigation', 'directions', 'move', 'rover navigation',
      'go to location',
    ],
  },
  {
    id: 'ROVER_SETTINGS',
    path: '/rover/settings',
    label: 'settings',
    roles: ['rover'],
    aliases: [
      'settings', 'preferences', 'configuration', 'my settings',
      'open settings', 'go to settings', 'settings page',
    ],
  },
  {
    id: 'ROVER_TALK',
    path: '/rover/talk',
    label: 'talk',
    roles: ['rover'],
    aliases: [
      'talk', 'communicate', 'speak', 'voice call', 'video call',
      'contact family', 'talk page',
    ],
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Route validation — checks that the path is a known registered route
// ─────────────────────────────────────────────────────────────────────────────
const ALL_VALID_PATHS = new Set(APP_ROUTES.map((r) => r.path));

export function isValidRoute(path: string): boolean {
  return ALL_VALID_PATHS.has(path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Similarity scoring (simple token-overlap based fuzzy match)
// ─────────────────────────────────────────────────────────────────────────────
function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9 ]/g, ' ')
      .split(/\s+/)
      .filter(Boolean),
  );
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  const aArr = Array.from(a);
  const bArr = Array.from(b);
  const intersection = aArr.filter((t) => b.has(t)).length;
  const union = new Set(aArr.concat(bArr)).size;
  return intersection / union;
}

// ─────────────────────────────────────────────────────────────────────────────
// Core resolve function
// ─────────────────────────────────────────────────────────────────────────────

export interface RouteResolutionResult {
  /** Resolved RouteEntry, or null if nothing found */
  route: RouteEntry | null;
  /** The alias that matched (for debugging) */
  matchedAlias: string | null;
  /** Similarity score 0–1 (1 = exact match) */
  score: number;
  /** Why the match was accepted or rejected */
  reason: string;
}

const FUZZY_THRESHOLD = 0.35; // minimum Jaccard score to accept a fuzzy match

/**
 * Resolves a free-form voice phrase to a registered route.
 *
 * Resolution priority:
 *  1. Exact alias match (case-insensitive)
 *  2. Contains-alias match  (phrase contains a known alias)
 *  3. Alias contains phrase (known alias contains the phrase)
 *  4. Fuzzy Jaccard token-overlap ≥ FUZZY_THRESHOLD
 *
 * @param phrase   - raw text from the voice command (e.g. "open settings")
 * @param role     - current user role for role-scoped disambiguation
 */
export function resolveRoute(
  phrase: string,
  role: UserRole = 'unknown',
): RouteResolutionResult {
  const normalised = phrase.toLowerCase().trim();

  // ── Remove common navigation prefixes so matching is on the noun ──────
  const cleaned = normalised
    .replace(/^(open|go to|navigate to|show me|take me to|i want to see|i want)\s*/i, '')
    .trim();

  // ── Role-scoped candidates first, then fall back to all routes ─────────
  const candidates =
    role !== 'unknown'
      ? APP_ROUTES.filter((r) => r.roles.includes(role) || r.roles.includes('unknown'))
      : APP_ROUTES;

  // 1. Exact alias match
  for (const entry of candidates) {
    for (const alias of entry.aliases) {
      if (alias === cleaned || alias === normalised) {
        return { route: entry, matchedAlias: alias, score: 1, reason: 'exact alias match' };
      }
    }
  }

  // 2. Phrase contains alias
  for (const entry of candidates) {
    for (const alias of entry.aliases) {
      if (cleaned.includes(alias) || normalised.includes(alias)) {
        return { route: entry, matchedAlias: alias, score: 0.9, reason: 'phrase contains alias' };
      }
    }
  }

  // 3. Alias contains phrase
  for (const entry of candidates) {
    for (const alias of entry.aliases) {
      if (alias.includes(cleaned) || alias.includes(normalised)) {
        return { route: entry, matchedAlias: alias, score: 0.8, reason: 'alias contains phrase' };
      }
    }
  }

  // 4. Fuzzy token-overlap
  const phraseTokens = tokenize(cleaned);
  let bestScore = 0;
  let bestEntry: RouteEntry | null = null;
  let bestAlias: string | null = null;

  for (const entry of candidates) {
    for (const alias of entry.aliases) {
      const score = jaccardSimilarity(phraseTokens, tokenize(alias));
      if (score > bestScore) {
        bestScore = score;
        bestEntry = entry;
        bestAlias = alias;
      }
    }
  }

  if (bestScore >= FUZZY_THRESHOLD && bestEntry) {
    return {
      route: bestEntry,
      matchedAlias: bestAlias,
      score: bestScore,
      reason: `fuzzy match (score=${bestScore.toFixed(2)})`,
    };
  }

  return { route: null, matchedAlias: null, score: bestScore, reason: 'no match found' };
}

/**
 * Returns a list of suggested route labels for fallback messaging.
 */
export function getSuggestions(phrase: string, role: UserRole, limit = 3): string[] {
  const candidates =
    role !== 'unknown'
      ? APP_ROUTES.filter((r) => r.roles.includes(role))
      : APP_ROUTES;

  const phraseTokens = tokenize(phrase);

  return candidates
    .map((entry) => {
      const best = Math.max(
        ...entry.aliases.map((a) => jaccardSimilarity(phraseTokens, tokenize(a))),
      );
      return { label: entry.label, score: best };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((e) => e.label);
}
