/**
 * NovaCare Voice Assistant — Intent Parser
 *
 * Converts raw speech transcripts into strongly-typed Intent objects.
 * Navigation intents are resolved through the centralized routeRegistry
 * instead of hardcoded strings, preventing 404 errors.
 */

import { resolveRoute, getSuggestions, UserRole, RouteResolutionResult } from './routeRegistry';

// ─────────────────────────────────────────────────────────────────────────────
// Intent types
// ─────────────────────────────────────────────────────────────────────────────

export type Intent =
  | { type: 'NAVIGATE'; resolution: RouteResolutionResult }
  | { type: 'CLICK'; target: string }
  | { type: 'SCROLL'; direction: 'up' | 'down' }
  | { type: 'SEARCH'; query: string }
  | { type: 'CONVERSATION'; text: string }
  | { type: 'UNKNOWN'; text: string };

// ─────────────────────────────────────────────────────────────────────────────
// Navigation trigger phrases
// ─────────────────────────────────────────────────────────────────────────────
const NAV_TRIGGERS = [
  'go to',
  'open',
  'navigate to',
  'show me',
  'take me to',
  'i want to see',
  'i want to go to',
  'load',
  'launch',
  'bring up',
  'go back',
  'go forward',
];

function isNavigationPhrase(text: string): boolean {
  return NAV_TRIGGERS.some((trigger) => text.includes(trigger));
}

// ─────────────────────────────────────────────────────────────────────────────
// Main intent parser
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @param text  - raw transcript from speech recognition
 * @param role  - current user role (used for role-aware route disambiguation)
 */
export function parseIntent(text: string, role: UserRole = 'unknown'): Intent {
  const lower = text.toLowerCase().trim();

  // ── Browser history navigation (special cases, no route registry needed) ─
  if (lower.includes('go back') || lower === 'back') {
    return {
      type: 'NAVIGATE',
      resolution: { route: null, matchedAlias: '__back__', score: 1, reason: 'browser back' },
    };
  }
  if (lower.includes('go forward') || lower === 'forward') {
    return {
      type: 'NAVIGATE',
      resolution: { route: null, matchedAlias: '__forward__', score: 1, reason: 'browser forward' },
    };
  }

  // ── Detect navigation intent ─────────────────────────────────────────────
  if (isNavigationPhrase(lower)) {
    const resolution = resolveRoute(lower, role);
    return { type: 'NAVIGATE', resolution };
  }

  // ── Even without trigger words, try route resolution (e.g. just "settings")
  const directResolution = resolveRoute(lower, role);
  if (directResolution.route !== null && directResolution.score >= 0.7) {
    return { type: 'NAVIGATE', resolution: directResolution };
  }

  // ── Click actions ────────────────────────────────────────────────────────
  if (lower.includes('click') || lower.includes('press') || lower.includes('tap')) {
    if (lower.includes('login') || lower.includes('log in') || lower.includes('sign in')) {
      return { type: 'CLICK', target: 'login' };
    }
    if (lower.includes('submit') || lower.includes('confirm')) {
      return { type: 'CLICK', target: 'submit' };
    }
    if (lower.includes('logout') || lower.includes('log out') || lower.includes('sign out')) {
      return { type: 'CLICK', target: 'logout' };
    }
    if (lower.includes('save')) return { type: 'CLICK', target: 'save' };
    if (lower.includes('cancel')) return { type: 'CLICK', target: 'cancel' };
  }

  // Logout / sign out without "click"
  if (lower.includes('logout') || lower.includes('log out') || lower.includes('sign out')) {
    return { type: 'CLICK', target: 'logout' };
  }

  // ── Scroll ───────────────────────────────────────────────────────────────
  if (lower.includes('scroll')) {
    if (lower.includes('down') || lower.includes('bottom')) {
      return { type: 'SCROLL', direction: 'down' };
    }
    if (lower.includes('up') || lower.includes('top')) {
      return { type: 'SCROLL', direction: 'up' };
    }
  }
  if (lower === 'down') return { type: 'SCROLL', direction: 'down' };
  if (lower === 'up') return { type: 'SCROLL', direction: 'up' };

  // ── Search ───────────────────────────────────────────────────────────────
  if (lower.startsWith('search for ') || lower.startsWith('find ') || lower.startsWith('search ')) {
    const query = lower
      .replace(/^search for\s+/, '')
      .replace(/^search\s+/, '')
      .replace(/^find\s+/, '')
      .trim();
    return { type: 'SEARCH', query };
  }

  // ── Conversational / greetings ───────────────────────────────────────────
  if (
    lower.startsWith('what') ||
    lower.startsWith('who') ||
    lower.startsWith('how') ||
    lower.startsWith('why') ||
    lower.startsWith('when') ||
    lower.startsWith('can you') ||
    lower.startsWith('hi') ||
    lower.startsWith('hello') ||
    lower.startsWith('hey') ||
    lower.includes('help me') ||
    lower.includes("what's") ||
    lower.includes("what is")
  ) {
    return { type: 'CONVERSATION', text };
  }

  return { type: 'UNKNOWN', text };
}

// ─────────────────────────────────────────────────────────────────────────────
// Response generator
// ─────────────────────────────────────────────────────────────────────────────

export function generateResponse(intent: Intent, role: UserRole = 'unknown'): string {
  switch (intent.type) {
    case 'NAVIGATE': {
      const alias = intent.resolution.matchedAlias;

      // Browser history specials
      if (alias === '__back__') return 'Going back.';
      if (alias === '__forward__') return 'Going forward.';

      const route = intent.resolution.route;
      if (!route) {
        // No route found — give helpful suggestions
        const suggestions = getSuggestions(alias ?? '', role);
        if (suggestions.length > 0) {
          return `I couldn't find that page. Did you mean: ${suggestions.join(', ')}?`;
        }
        return "I couldn't find that page. Please try again with a different phrase.";
      }
      return `Opening ${route.label}.`;
    }

    case 'CLICK':
      return `Triggering ${intent.target}.`;

    case 'SCROLL':
      return `Scrolling ${intent.direction}.`;

    case 'SEARCH':
      return `Searching for "${intent.query}".`;

    case 'CONVERSATION': {
      const lower = intent.text.toLowerCase();
      if (lower.includes('hello') || lower.includes('hi') || lower.includes('hey')) {
        return "Hello! I'm Nova, your AI assistant. How can I help you today?";
      }
      if (lower.includes('what can you do') || lower.includes('help')) {
        return "I can navigate pages, scroll, search, and answer questions. Try saying 'open settings' or 'go to appointments'.";
      }
      return "I'm here to help. I can navigate the app, search, and assist with tasks.";
    }

    case 'UNKNOWN':
      return "I heard you, but I'm not sure how to help with that. Try saying 'open settings' or 'go to dashboard'.";
  }
}
