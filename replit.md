# CallShield

CallShield reviews call recordings and transcripts for spam signals, exposes model evaluation metrics, and makes uncertainty visible to the reviewer.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/spam-call-detector/` — React/Vite web app with analyze and model bench routes
- `artifacts/api-server/src/routes/detection.ts` — transcription, prediction, metrics, and dashboard API
- `lib/api-spec/openapi.yaml` — source of truth for the API contract
- `ml_pipeline/` — Python training, preprocessing, Whisper inference, and setup notes
- `artifacts/spam-call-detector/src/index.css` — CallShield theme tokens and visual language

## Architecture decisions

- The browser uses the generated API client from the OpenAPI contract; the API remains stateless because model artifacts are trained locally and can be swapped in later.
- The API uses a real Faster-Whisper worker for audio transcription and loads a persisted trained classifier when `ml_pipeline/artifacts/callshield.joblib` exists; the heuristic path remains an explicit fallback.
- Production ML work lives in `ml_pipeline/`: `train.py` compares all eight requested classifiers and persists the vectorizer plus recommended model with joblib.
- Uploaded audio is accepted as multipart, written to a temporary file for transcription, and deleted after the Faster-Whisper worker returns timestamped segments.

## Product

- Analyze a pasted transcript or audio recording
- Capture audio from a microphone, preview it, and inspect an upload waveform
- See spam/not-spam assessment, confidence, detected signals, and engineered features
- Compare all eight classifiers with accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, latency, training time, and confusion matrices
- Retrain with `ml_pipeline/train.py`, then run local audio inference with selectable Whisper model sizes

## User preferences

_No project-specific preferences recorded._

## Gotchas

- The first transcription with a new Whisper size downloads its model weights; if the runtime is unavailable, the API returns a setup error instead of fake transcript text.
- The current trained bundle was built from the supplied 800-row CSV using `text` and `is_spam`; `is_confidential` remains available for a future separate classifier.
- Run `pnpm --filter @workspace/api-spec run codegen` after changing `lib/api-spec/openapi.yaml`.
- The web artifact workflow supplies `PORT` and `BASE_PATH`; do not run the Vite command manually without those environment variables.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
