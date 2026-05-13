NovaCare — Migration Map (Phase 2 planning)

Purpose: safe, stepwise plan to reorganize the repository into a clean professional structure without breaking functionality. All moves are copy-first; originals remain until verification and CI pass. For moves that change import paths, we will add compatibility shims at original locations.

High-level notes:
- No destructive deletions in Phase 2. Files only moved to `archive/` when uncertain. Clear duplicates will be archived first, then optionally deleted after verification.
- Secrets (PEM files, .env with real credentials) will be removed from repo and moved to `archive/secrets_removed/` or a local secure store; replaced with `.env.example` files.

PHASE 2 — Planned Copies & Moves (copy -> verify -> remove original)
------------------------------------------------------------------
1) Documentation consolidation
   - Copy: `docs/*` -> `docs/architecture/`, `docs/setup/`, `docs/APIs/` as appropriate
   - Copy: `optimized_runtime/*.md` -> `docs/deployment/optimized_runtime/`
   - Copy: `services/*/README.md` -> `docs/services/<service>/README.md` (keep service README.md too)
   - Move (if duplicate): `README.md` contents will be reconciled into root README and `docs/`

2) Apps
   - Copy: `frontend/` -> `apps/frontend/` (preserve .env and .env.local as templates only)
   - Copy: `novacare_app/` -> `apps/mobile/`
   - Copy: `optimized_runtime/robot_ui` -> `apps/robot_ui/` (if desired)
   - At root: leave small wrapper scripts to run apps from original positions until migration completed

3) Services
   - Keep existing `services/` directory as canonical location. For example:
       services/llm-backend/  (canonical)
       services/asl-model/    (canonical)
       services/robot/        (canonical)
   - Detect duplicates:
       - Root `LLMs/` -> Proposed: `services/llm-backend/LLMs/` (consolidate into one place)
       - Root `Emotion_Detection/` -> Proposed: `services/emotion/` or `shared/models/emotion/`
   - Action: copy root AI module duplicates into `shared/models/` or the canonical service; create duplicate_report.md

4) Shared
   - Move common helpers and utils into `shared/utils/`.
       - e.g., `utils/`, `Conversation/` helpers, duplicated `speech.ts`/`novabot-api` logic
   - Copy `static/js/*` duplicates into `shared/assets/js/` and update services to reference shared assets (stepwise)

5) Infrastructure
   - Copy all Dockerfiles into `infrastructure/docker/` (keep service-level Dockerfiles until validated)
   - Copy `deploy/*` -> `infrastructure/deployment/` (move ssh keys to archive/secrets_removed/)
   - Move start scripts: `start_all.sh`, `start-novacare.sh`, `start_all.*` -> `infrastructure/scripts/`
   - Add root shim scripts to call new locations

6) Tests
   - Copy root test files `test_*.py` -> `tests/experimental/` with metadata about origin
   - Copy `services/*` tests -> `tests/integration/` or `tests/AI/` as appropriate
   - Update CI later to reference tests/ directory

7) Archive (immediate candidates)
   - Move suspicious or obviously duplicate files to `archive/` with metadata in `archive/README.md`:
       - `grad_project___proposal (2).pdf` -> `archive/docs/`
       - `project book part 1.pdf` -> `archive/docs/`
       - `serbot manual.pdf` -> `archive/docs/`
       - `generate_tree.py` (if only used for snapshot) -> `archive/tools/` (confirm usage first)
       - `deploy/frontend-novacare.pem` -> `archive/secrets_removed/` (remove from active repo)

8) Naming normalization
   - Standardize service directories to lowercase hyphen-separated names where practical (e.g., `llm-backend` → `llm-backend` kept)
   - Normalize file names: use snake_case for python modules, kebab-case for scripts, PascalCase for React components
   - Create `naming_normalization_report.md` listing changes and rationale

9) Compatibility shims
   - For each Python module moved, create a small shim in original location: e.g., in `Conversation/conversation_ai.py` create `from shared.utils.conversation import *` to preserve imports until tests updated.
   - For JS/TS assets, create a forwarding import or copy until updated.

10) CI and verification
   - After copies, run unit tests and start basic services locally in debug mode (manual/automated).
   - Validate Docker build for a single service (llm-backend) from both original Dockerfile and `infrastructure/docker/` copy.

MIGRATION PHASES & TIMELINE (suggested)
----------------------------------------
Phase A (safe copies + docs) — low risk (1 day)
  - Create new directories
  - Copy docs and NAVIGATION_GUIDE
  - Copy apps into apps/
  - Create archive/ and move clear junk

Phase B (tests + shared utilities) — moderate risk (1—2 days)
  - Consolidate tests into tests/
  - Copy utils into shared/utils/
  - Create duplicate_code_report.md

Phase C (service canonicalization + shims) — higher risk (2—4 days)
  - Consolidate LLM modules into single canonical service
  - Add compatibility shims for moved modules
  - Update a small number of imports and validate

Phase D (cleanup + remove duplicates) — final (after validation, 1—2 days)
  - Remove archived duplicates only after verification
  - Update CI to use new test paths
  - Finalize root README

Notes on imports and runtime
----------------------------
- Many Python imports rely on file locations. To avoid breakage, after copying modules to new locations we will *not* delete originals until shims are in place and tests pass.
- For Node/Next.js, environment variables and build outputs must be preserved.

Deliverables from Phase 2 (after sign-off)
------------------------------------------
- `MIGRATION_MAP.md` (this file)
- `PHASE2_ACTIONS.md` — full list of copy commands and shims to be applied
- `duplicate_code_report.md` — identified duplicates and suggested canonical locations
- `archive/` folder with moved items and `archive/README.md` explaining reasons
- `naming_normalization_report.md`

Approval
--------
Reply with `APPROVE MIGRATION MAP` to proceed to Phase 2 (copy operations). Reply with `FEEDBACK` and include specific constraints if you want some directories left untouched or certain files prioritized.
