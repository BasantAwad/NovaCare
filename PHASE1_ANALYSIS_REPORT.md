NovaCare — Phase 1: Full Repository Analysis

Date: 2026-05-13
Scope: Deep inventory and identification of duplicates, scattered tests, documentation spread, orphaned assets, and potential cleanup candidates. This report catalogs findings and recommended safe actions (no deletions yet).

SUMMARY
-------
The repository is functionally complete but contains multiple duplicated folders, scattered tests and scripts, many project-specific docs across the tree, multiple frontends (Next.js + Flutter), duplicate AI modules, and numerous root-level utilities and startup scripts. Some directories are clearly service boundaries (services/*) but there are also several parallel copies (LLMs/, services/llm-backend/LLMs). There are many README.md files spread across services.

OBJECTIVES
----------
- Identify duplicates and likely dead/experimental files
- Propose a safe migration map to a normalized structure
- Mark files to archive (rather than delete) where uncertain
- Provide a prioritized next-step plan for refactoring

FINDINGS (high-level)
---------------------
- Root-level clutter: many scripts, tests, and docs live at repo root. Examples: start_all.sh, start-novacare.sh, start_all.ps1, start_all.bat, HOW_TO_RUN.md, AXIOM_SETUP.md, manual.txt, project book PDFs.
- Multiple README.md files in service subfolders (good) plus multiple top-level docs/ folder and optimized_runtime docs — documentation is scattered.
- Tests are scattered in root and services; several `test_*.py` files exist at root: test_axiom.py, test_chat.py, test_robot.py, test_lidar_hardware.py, test_axiom_simple.py.
- Duplicate/overlapping AI code: `LLMs/` directory at root and `services/llm-backend/LLMs/`. Also `frontend/ai/impl/conversational_ai.py` and `LLMs/conversational_ai.py` both contain conversational AI logic — likely duplication.
- Frontend duplication: `frontend/` (Next.js) and `src/` top-level (appears to be an extracted frontend copy) and `novacare_app/` (Flutter mobile). There are overlapping `static/js/` and `services/llm-backend/static/js/` assets (NovaBotClient.js, STT.js, TTS.js) — duplicate assets.
- Multiple Dockerfiles across services: services/robot/Dockerfile, services/asl-model/Dockerfile, services/llm-backend/Dockerfile, frontend/Dockerfile. Keep them but centralize in infrastructure/docker/ once validated.
- Multiple start scripts: `start_all.sh`, `start-novacare.sh`, `start_all.*` and `scripts/start_robot.sh` and `optimized_runtime/scripts/startup.sh`. Consolidate under `infrastructure/scripts/` with clear naming.
- `deploy/` contains jetson-specific services and keys. This belongs in `infrastructure/deployment/` with secure handling of secrets (do not commit credentials). Verify `frontend-novacare.pem` — if it's a private key it must be archived securely or moved to secure store. Do NOT leave private keys in repo.
- `docs/` exists at root with many docs (architecture.md, API_Services_Documentation.md, etc). Also `optimized_runtime` contains its own comprehensive docs. Consolidate all docs under `docs/` with subfolders for architecture, deployment, services, integration.
- `utils/`, `Conversation/`, and small modules may contain helpers duplicated across services. Need targeted import analysis.
- `generated` or build artifacts: `tsconfig.tsbuildinfo`, `package-lock.json`, `pubspec.lock`, some PDFs — move non-source into `archive/` or ignore via .gitignore.

AUTOMATED SEARCH HIGHLIGHTS
---------------------------
- test files found:
  - test_axiom_simple.py (root)
  - test_axiom.py (root)
  - test_lidar_hardware.py (root)
  - test_chat.py (root)
  - test_robot.py (root)
  - services/asl-model/test_api.py
  - services/llm-backend/test_mental_health_pipeline.py
  - services/llm-backend/test_api_alternatives.py

- README files found: multiple (root and per-service). Consolidation recommended but keep per-service README.md (update them).
- Dockerfiles found: services/robot, services/asl-model, services/llm-backend, frontend
- Start scripts found: root start_all.*, scripts/jetson/start_robot.sh, optimized_runtime/scripts/startup.sh

RISK ITEMS (must be handled carefully)
-------------------------------------
- `frontend/.env`, `frontend/.env.local`, `.env` files — ensure secrets are not leaked.
- `deploy/frontend-novacare.pem` — likely a private key. Move to secure storage immediately; remove from repo if sensitive.
- Files under `novacare_db.sql` and migration scripts — these are important for DB; DO NOT alter.
- `services/robot/robot_hal.py` and `robot_service.py` — hardware-facing code. Preserve as-is.
- Anywhere `import` paths assume files at current locations; moving Python modules will break imports unless we update package names and paths.

RECOMMENDATIONS (safe, non-destructive approach)
------------------------------------------------
1. Create new top-level directories (do not delete anything yet):
   - `apps/` (for frontends and mobile apps)
   - `services/` (keep existing service subfolders here)
   - `shared/` (for shared utils, models, configs)
   - `infrastructure/` (for docker, deployment, scripts)
   - `docs/` (single place for all documentation)
   - `tests/` (consolidate tests)
   - `archive/` (for uncertain/legacy files)

2. Copy (not move) files into the new structure for initial validation; leave originals until everything is updated and CI passes.

3. Consolidate documentation under `docs/` with subfolders: `architecture/`, `deployment/`, `setup/`, `APIs/`, `hardware/`, `research/`.

4. Consolidate tests: copy root test_*.py into `tests/experimental/` and move service-specific tests into `tests/integration/` or `tests/AI/` accordingly.

5. Identify and catalog duplicated modules (e.g., conversational_ai.py duplicates). Create a `duplicate_code_report.md` listing duplicates with recommended canonical location.

6. For file moves that will change imports, implement migration in two steps:
   - Create a compatibility shim in the old location that imports from the new location (thin wrapper) so imports won't break until we fully refactor.
   - Update imports in codebase to new paths, run tests, then remove shim.

7. Do NOT remove any Dockerfiles or service files until the new structure is validated and compose scripts updated.

8. For secrets in repo (PEM keys, .env with secrets), immediately move them to `archive/secrets_removed/` and replace with `.env.example` templates. Mark this as a security cleanup priority.

9. Add `.gitignore` entries for build artifacts and compiled caches (tsbuildinfo, *.pyc, build/, .venv/, .cache/).

PROPOSED HIGH-LEVEL TARGET STRUCTURE
-----------------------------------
project_root/
├── apps/
│   ├── frontend/       # Next.js app content (moved from frontend/)
│   ├── mobile/         # Flutter app (novacare_app/)
│   └── robot_ui/       # optional lightweight UI (optimized_runtime/robot_ui)
│
├── services/
│   ├── auth/           # services/auth-backend
│   ├── llm/            # services/llm-backend
│   ├── asl/            # services/asl-model
│   ├── emotion/        # Emotion_Detection
│   ├── speech/         # edge-tts-proxy, etc.
│   └── robot_runtime/  # optimized_runtime or services/robot
│
├── shared/
│   ├── models/         # ML model wrappers
│   ├── utils/          # reusable utilities (utils/, shared code)
│   ├── protocols/      # pydantic schemas, API contracts
│   ├── configs/        # central config management
│   └── assets/         # shared images, icons
│
├── infrastructure/
│   ├── docker/         # dockerfiles and compose
│   ├── deployment/     # deploy scripts & cloud configs
│   ├── monitoring/     # health & monitoring configs
│   └── scripts/        # start/stop scripts
│
├── docs/
│   ├── architecture/
│   ├── deployment/
│   ├── hardware/
│   ├── APIs/
│   ├── setup/
│   └── troubleshooting/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── hardware/
│   ├── AI/
│   └── experimental/
│
├── archive/            # legacy / uncertain files
└── README.md

DELIVERABLES (Phase 1)
----------------------
- This analysis report (PHASE1_ANALYSIS_REPORT.md)
- Migration map (MIGRATION_MAP.md) — next
- Prioritized action items

PRIORITIZED ACTIONS (safe order)
--------------------------------
1. Create new directories under repo root: apps, shared, infrastructure, docs, tests, archive.
2. Copy docs from existing `docs/`, `optimized_runtime/`, `services/*/README.md` into `docs/` subfolders.
3. Copy `frontend/` into `apps/frontend/` and `novacare_app/` into `apps/mobile/` (keep originals until validated).
4. Copy optimized_runtime into `services/robot_runtime/` (or keep under optimized_runtime for now).
5. Move root-level scripts to `infrastructure/scripts/` (copy first, add shim wrappers at root to avoid breaking start scripts).
6. Consolidate tests under `tests/` (copy, leave originals).
7. Create a `duplicate_code_report.md` identifying conversational_ai duplicates and other candidates for consolidation.
8. Identify secrets (PEM, .env containing keys) and move them to `archive/secrets_removed/` while replacing with `.env.example`.
9. Add top-level `README.md` describing new structure and pointing to NAVIGATION_GUIDE.md.

NEXT STEPS
----------
- I will create a migration map file (`MIGRATION_MAP.md`) listing planned file moves and copies, grouped by phase. This will not execute any moves yet. The map will include safe copy-and-shim strategy for imports.
- After you review and approve the map, I will implement Phase 2 (identify duplicates and move unambiguous files into archive/). All moves will be copy-first to avoid breaking the running system.

END OF REPORT
