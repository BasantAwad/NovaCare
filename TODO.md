# NovaCare Professionalization Refactor — Progress Tracker

> Safety policy: **No functional logic rewrite**. **No deletions** unless clearly obsolete/unused. Prefer **copy-first + index + shim**.

## Phase 1 — Full codebase analysis
- [x] Inventory: checked root structure and key entrypoint/control files
- [ ] Deepen scan: identify duplicates and import risks across Python/TS

## Phase 2 — Create normalized professional structure (non-destructive)
- [ ] Create scaffold directories: `apps/ shared/ infrastructure/ docs/ tests/ archive/` (copy-first strategy)


- [ ] Add `docs/README.md` index + standardized doc categories
- [x] Add `docs/templates/` (architecture doc template + service readme template)

- [ ] Add `tests/` category folders + initial `README.md` (no test moves yet)
- [ ] Add `duplicate_code_report.md` (report-only): identify likely duplicates (LLM wrappers, conversational_ai, asset JS)
- [ ] Add `dead_code_report.md` (report-only): identify candidates for archive/legacy

## Phase 3 — README standardization (after Phase 2)
- [ ] Update root `README.md` to be strict entrypoint + links
- [ ] Update service READMEs to match checklist

## Phase 4 — Test consolidation (copy-first)
- [ ] Create `pytest.ini`
- [ ] Copy root `test_*.py` into `tests/experimental/`
- [ ] Copy service tests into proper `tests/AI`/`tests/integration`

## Phase 5 — Naming normalization + migration map execution
- [ ] Write naming rules doc: `docs/design_guidelines/naming.md`
- [ ] Implement migration map using shims; run smoke tests

## Phase 6 — Archive system (only after confidence is high)
- [ ] Move safe obsolete/duplicate/generated artifacts into `archive/`
- [ ] Move secrets (if any found) into `archive/secrets_removed/`

## Phase 7 — Quality gates
- [ ] Run unit/integration tests
- [ ] Run basic service smoke checks (where possible without hardware)
- [ ] Validate imports after any moves

