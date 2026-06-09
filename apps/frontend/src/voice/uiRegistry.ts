/**
 * NovaCare Nova AI — UI Element Registry
 *
 * Components and pages self-register their interactive elements here.
 * Nova looks up this registry before scanning the live DOM, giving
 * priority to explicitly registered elements.
 *
 * Registration is automatic: use the `useVoiceElement` hook inside any
 * component — no changes to VoiceContext or intentParser required.
 */

export type UIElementType =
  | 'button'
  | 'link'
  | 'input'
  | 'select'
  | 'modal-trigger'
  | 'modal-close'
  | 'tab'
  | 'dropdown'
  | 'toggle'
  | 'form'
  | 'menu-item'
  | 'action';

export interface UIElement {
  /** Unique stable ID */
  id: string;
  /** Primary human label (used for TTS confirmation) */
  label: string;
  /** All phrases that should trigger this element */
  aliases: string[];
  /** Element type for context-aware matching */
  type: UIElementType;
  /** The action to perform */
  action: () => void;
  /** Optional: element is only active on specific pathnames */
  scope?: string | RegExp;
  /** Priority — higher wins over lower when aliases clash (default 0) */
  priority?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry store
// ─────────────────────────────────────────────────────────────────────────────

class UIRegistry {
  private elements: Map<string, UIElement> = new Map();

  register(element: UIElement): void {
    this.elements.set(element.id, {
      priority: 0,
      ...element,
      aliases: element.aliases.map((a) => a.toLowerCase().trim()),
    });
  }

  unregister(id: string): void {
    this.elements.delete(id);
  }

  /** Returns all currently registered elements */
  getAll(): UIElement[] {
    return Array.from(this.elements.values());
  }

  /** Returns elements whose scope matches the given pathname */
  getForPath(pathname: string): UIElement[] {
    return this.getAll().filter((el) => {
      if (!el.scope) return true;
      if (typeof el.scope === 'string') return pathname.startsWith(el.scope);
      return el.scope.test(pathname);
    });
  }
}

export const uiRegistry = new UIRegistry();
