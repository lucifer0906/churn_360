# Churn360: Customer Churn and Retention Intelligence
## MLOps Course Project — Build Specification for AI-Assisted Development

---

## 0. HOW TO USE THIS DOCUMENT (read this first, AI assistant)

This is a staged build plan. **Do not attempt to build the entire project in one shot.**
Work through the stages in order. At the end of each stage:
1. Stop and show what was built.
2. Confirm it actually runs (tests pass / command succeeds / endpoint responds).
3. Wait for explicit confirmation before moving to the next stage.

Do not skip ahead to later-stage tools (e.g. do not add Airflow in Stage 1, do not add
Docker before the API works locally). Each stage assumes only the previous stages exist.
If something in a later stage seems to require redesigning an earlier stage, flag it and ask
rather than silently refactoring.

Where a step requires something outside your execution environment (AWS account, GitHub
repo creation, pushing secrets, Grafana cloud setup, etc.), stop and give the user clear
manual instructions instead of attempting it yourself.

---

## 1. PROJECT CONTEXT

This is a course project for an **MLOps** subject. The professor has given a list of 18
possible industry-style project topics, each requiring the **same mandatory end-to-end
MLOps architecture**. The ML task itself (what you're predicting) is the *easy* part of
this assignment — the point of the assignment is the pipeline: versioning, experiment
tracking, orchestration, deployment, monitoring, CI/CD, and responsible AI documentation.

**A notebook that only trains a model and reports accuracy is explicitly NOT an acceptable
submission**, even if the model is good. The grade is on the plumbing, not the model quality.

### 1.1 Chosen project: Churn360

- **Industry problem**: A subscription/telecom business wants to identify customers who are
  likely to cancel ("churn") before they do, so retention offers can be made proactively.
- **ML task**: Binary classification — predict `churn` (yes/no) from customer attributes
  (tenure, monthly charges, contract type, support calls, etc.)
- **Scoring pattern**: Monthly **batch scoring** (not real-time streaming) — simpler to
  pipeline and demo than a low-latency real-time system.
- **Why this project was chosen over the other 17 options**: tabular data (no CV/NLP
  complexity), a very standard and clean public dataset exists (Telco Customer Churn),
  SHAP works cleanly on tree-based models for tabular data, and — unlike HealthWatch,
  EduRisk, FairLend, or TalentMatch — this project has **no explicit fairness/subgroup
  analysis requirement** forced by the brief, keeping the Responsible AI section simpler
  (still do a basic SHAP explanation + model card + data card, just not subgroup fairness
  testing).

### 1.2 Dataset

Use the **IBM Telco Customer Churn dataset** (public, available on Kaggle and in several
GitHub mirrors). ~7,000 rows, one row per customer, with a `Churn` label column. Do not
substitute a different dataset without discussion — this one is clean, well-documented,
and has enough reference material online that implementation questions can be resolved
quickly.

---

## 2. THE MENTAL MODEL — 5 BOXES

Before writing code, hold this mental model. Everything mandated by the professor is a tool
that belongs inside exactly one of these five boxes. Do not conflate them or build them out
of order.

```
BOX 1: DATA           BOX 2: TRAINING         BOX 3: SERVING          BOX 4: MONITORING
(get + clean the       (train models,          (live API that          (watch the API,
 churn data)       →    track experiments,  →   answers "will       →   detect if incoming
                        register best one)       this customer           data looks
                                                  churn?")                different now)
                             ↕
                        BOX 5: AUTOMATION
                   (Airflow runs Boxes 1–2 on schedule;
                    GitHub Actions tests code on every push)
```

| Box | Responsibility | Tools |
|---|---|---|
| 1. Data | ingest, validate, clean, feature-engineer | DVC |
| 2. Training | train baseline + 2 candidates, log everything, register winner | MLflow |
| 3. Serving | wrap the registered model in an API | FastAPI + Docker |
| 4. Monitoring | track live API health + data drift | Prometheus + Grafana |
| 5. Automation | orchestrate Boxes 1–2 on schedule; test on every commit | Airflow + GitHub Actions |

Cutting across all boxes: **SHAP explanations** (surfaced in Box 3's API output) and
**documentation** (model card, data card, responsible-AI statement — written once
everything else is built, since you need to know what you built to document it honestly).

---

## 3. MANDATORY COMPONENTS (from the professor's brief — do not omit any)

1. **Version control and reproducibility**: Git + GitHub, DVC for dataset/model versioning,
   fixed random seed, requirements/environment lock file.
2. **Experiment management**: at least 1 baseline + 2 candidate models, all params/metrics/
   artifacts logged in MLflow, best model registered with version + alias + metadata.
3. **Automated ML workflow**: Airflow DAG covering ingestion → validation → preprocessing →
   feature engineering → training → evaluation → model registration.
4. **Deployment**: FastAPI REST service, input schema validation, error handling,
   health-check + prediction endpoints, Dockerfile + Docker Compose.
5. **CI/CD and testing**: GitHub Actions workflow, unit tests for preprocessing + prediction,
   API integration test, Docker image build test, model-quality threshold gate before
   promotion.
6. **Monitoring**: request count, error rate, inference latency, prediction distribution,
   data/feature drift, model performance once labels are available, Prometheus metrics +
   Grafana dashboard.
7. **Responsible AI**: SHAP explanation, model card, data card, privacy/ethical-limitations/
   intended-use statement. (No subgroup fairness test required for this specific project,
   unlike HealthWatch/EduRisk/FairLend/TalentMatch — but do still note in the model card
   that churn predictions should be used as decision support, not as an autonomous action,
   as general good practice even though this project isn't in the professor's explicit
   "decision-support only" list of Projects 7/15/17.)
8. **Cloud extension**: AWS S3 for data/artifact storage, SageMaker training job or
   deployment, CloudWatch logging. **Tear down all cloud resources immediately after the
   demo** to avoid cost.

### Required final submission checklist
- GitHub repo with meaningful commit history
- Professional README with architecture diagram + one-command setup
- DVC metadata and data-version evidence
- MLflow experiment + model-registry screenshots
- Airflow DAG source + successful-run screenshot
- FastAPI Swagger/OpenAPI screenshot
- Dockerfile + Docker Compose file
- Successful GitHub Actions workflow run
- Monitoring dashboard + simulated-drift evidence
- Data card, model card, responsible-AI statement
- Performance report: model metric, business metric, p95 inference latency, throughput,
  error rate, drift metric
- Five-minute live demo
- Individual viva

---

## 4. BUILD ORDER — WHY THIS SEQUENCE

Do **not** build in the order the requirements are listed above (Git → DVC → MLflow →
Airflow → FastAPI → ...). That order sets up automation before there's anything worth
automating, and sets up monitoring before there's a live service to monitor. Build in
this order instead: **get something working manually first, then progressively wrap
tooling around it.** Each stage below only adds one new layer, and only after the
previous layer is confirmed working standalone.

---

## 5. STAGED BUILD PLAN

### STAGE 1 — Manual baseline (no MLOps tooling at all)
**Goal**: A plain Python script, run top to bottom, that loads the Telco churn CSV, cleans
it, engineers a few basic features, trains three models (Logistic Regression as baseline,
Random Forest and XGBoost as candidates), and prints accuracy/precision/recall/F1 for each.

**Do:**
- Set up the repo folder structure (see Section 6).
- `requirements.txt` with pinned versions.
- Fixed random seed everywhere (`random_state=42` or similar, used consistently).
- A `src/data/load_and_clean.py` and `src/train/train.py` (or similar) — not notebook cells.
  Notebooks are fine for exploration but the actual pipeline must be `.py` scripts.
- No MLflow, no Docker, no Airflow yet. Just confirm the ML task itself works end to end.

**Stop condition**: running `python src/train/train.py` trains all 3 models and prints
metrics without errors.

---

### STAGE 2 — Experiment tracking (MLflow)
**Goal**: Wrap the Stage 1 training script with MLflow logging. Do not rewrite the training
logic — add tracking calls around what already works.

**Do:**
- `mlflow.start_run()` around each of the 3 model trainings.
- Log params (model type, hyperparameters), metrics (accuracy, precision, recall, F1,
  ROC-AUC), and the model artifact itself.
- After all 3 runs, register the best-performing model in the MLflow Model Registry with a
  version and an alias (e.g. `champion` or `production`).
- Run MLflow's local tracking UI to confirm all 3 runs are visible and comparable.

**Stop condition**: `mlflow ui` shows 3 logged runs with metrics, and one registered model
version exists.

---

### STAGE 3 — Serving API (FastAPI, no Docker yet)
**Goal**: Load the registered model from Stage 2 and serve predictions over HTTP, running
directly on the host machine (no containerization yet).

**Do:**
- `/predict` endpoint: accepts customer features (defined via a Pydantic schema — this
  gives input validation for free), returns churn prediction + probability.
- `/health` endpoint: simple liveness check.
- Basic error handling for malformed input (Pydantic will largely handle this; add explicit
  handling for anything it won't, e.g. out-of-range categorical values).
- Test locally via FastAPI's auto-generated Swagger UI (`/docs`).

**Stop condition**: hitting `/predict` with a sample customer payload via the Swagger UI
returns a sane prediction, and `/health` returns 200.

---

### STAGE 4 — Containerization (Docker)
**Goal**: Package the working Stage 3 API into a container. Do not change any application
code in this stage — only packaging.

**Do:**
- `Dockerfile` for the FastAPI service.
- `docker-compose.yml` — even with just one service at this stage, structure it so later
  stages (Prometheus, Grafana, Airflow) can be added as additional services in the same
  file rather than a rewrite.
- Confirm the containerized API responds identically to the Stage 3 host-run version.

**Stop condition**: `docker compose up` brings up the API, and `/predict` + `/health` work
identically to Stage 3.

---

### STAGE 5 — Data versioning (DVC) + GitHub formalization
**Goal**: Version the dataset with DVC, and make sure the GitHub repo is in proper shape
(this should have had commits all along, but formalize structure/README here).

**Do:**
- `dvc init`, track the raw CSV and any processed data artifacts.
- Push DVC-tracked data to a remote (local remote is fine for a course project if cloud
  storage isn't set up yet — see Stage 9 for S3 as the eventual remote).
- Confirm `dvc status` / `dvc pull` on a fresh clone correctly retrieves the versioned data.
- Clean, meaningful commit history from this point forward — this is graded.

**Stop condition**: a fresh clone of the repo + `dvc pull` reproduces the exact dataset
used in Stage 1–4.

---

### STAGE 6 — Orchestration (Airflow DAG)
**Goal**: Wire the already-working Stage 1–2 steps (data load → clean → feature engineer →
train → evaluate → register) into an Airflow DAG, so it runs as a scheduled/triggerable
pipeline instead of manually running scripts.

This is the first genuinely new concept if you (Chetan) haven't used Airflow before. Expect
this to take longer than the earlier stages even though the underlying logic already works —
the new part is learning how DAGs, tasks, and the Airflow scheduler work, not the ML logic.

**Do:**
- One task per pipeline step (ingestion, validation, preprocessing, feature engineering,
  training, evaluation, model registration) — matching the professor's explicit list.
- Use the Airflow local/standalone setup (not a production cluster) — appropriate for a
  course project.
- Trigger a manual DAG run and confirm all tasks succeed in order.

**Stop condition**: a full DAG run, triggered from the Airflow UI, completes all tasks
green, and produces a newly registered MLflow model version as its end result.

---

### STAGE 7 — Monitoring (Prometheus + Grafana)
**Goal**: Instrument the FastAPI service (from Stage 4) with metrics, and visualize them.

**Do:**
- Add a metrics endpoint to FastAPI (e.g. via `prometheus-fastapi-instrumentator` or
  similar) exposing request count, error rate, and latency.
- Add Prometheus as a service in `docker-compose.yml`, scraping the FastAPI metrics
  endpoint.
- Add Grafana as a service, connect it to Prometheus as a data source, build a dashboard
  showing request count, p95 latency, error rate.
- **Simulate drift**: deliberately send the API a batch of inputs with a shifted
  distribution (e.g. unusually high monthly charges across the board) and show this
  reflected in a drift metric/panel. A simple statistical drift check (e.g. population
  stability index or a KS-test on a couple of key features, computed periodically and
  pushed to Prometheus as a custom metric) is sufficient for course-project scope — this
  doesn't need to be a full drift-detection framework.

**Stop condition**: Grafana dashboard shows live request metrics, and a before/after
screenshot pair demonstrates the drift metric reacting to the simulated shifted input batch.

---

### STAGE 8 — CI/CD (GitHub Actions)
**Goal**: Automated testing and build-checking on every push. This should be quick, since
everything it tests already works by this point.

**Do:**
- Unit tests: preprocessing functions, prediction function (mock or small fixture data is
  fine — don't require the full dataset in CI).
- API integration test: spin up the FastAPI app in the test run, hit `/predict` and
  `/health`, assert sane responses.
- Docker image build test: confirm the image builds successfully in CI.
- Model-quality gate: a check step that fails the workflow if a newly trained model's
  metric falls below a defined threshold (ties back into Stage 6's registration step —
  this is where "don't promote a worse model" gets enforced).
- GitHub Actions workflow file (`.github/workflows/ci.yml`) running on push/PR.

**Stop condition**: a push to the repo triggers the Actions workflow, and it completes
green (or correctly fails/blocks promotion if a model is deliberately degraded for testing).

---

### STAGE 9 — Cloud extension (AWS) — do this last, and only briefly
**Goal**: Demonstrate the pipeline *can* run on real cloud infrastructure, without needing
to permanently host it there.

**I (the AI assistant) must guide Chetan through this stage interactively and NOT attempt
AWS console actions on his behalf** — AWS account setup, IAM permissions, and billing are
things he needs to do himself with guidance, step by step, confirming each before moving
to the next. Suggested flow:

1. Create an S3 bucket; point DVC's remote at it (replacing/adding to Stage 5's remote).
   Upload data/artifacts there.
2. Either (a) run a SageMaker training job using the Stage 1–2 training logic, or
   (b) deploy the registered model to a SageMaker endpoint — pick whichever is simpler
   given time remaining; a training job is usually the lower-effort option for a course
   demo.
3. Point CloudWatch at whatever was deployed, for basic logging visibility.
4. **Immediately after the live demo**: delete the SageMaker endpoint/training resources
   and empty/delete the S3 bucket (or at minimum stop any billable running resources).
   Confirm this explicitly — don't leave this for "later," it's easy to forget and
   AWS costs accumulate silently.

**Stop condition**: a screenshot/demo of data in S3 and a completed SageMaker job/endpoint
exists for submission evidence, and cloud resources are torn down within the same session
as the demo.

---

### STAGE 10 — Responsible AI documentation + final packaging
**Goal**: Write the documentation that can only honestly be written once the system exists.

**Do:**
- **SHAP integration**: add SHAP explanation output to the `/predict` response (or a
  separate `/explain` endpoint) — for tree-based models (RF/XGBoost) this is
  straightforward with `TreeExplainer`.
- **Model card**: model type, training data summary, metrics, intended use (decision
  support for retention offers — not an autonomous cancel/retain decision), known
  limitations.
- **Data card**: dataset source, size, feature descriptions, any known biases or gaps.
- **Responsible-AI statement**: privacy notes (this is customer data — note what would be
  needed for real deployment, e.g. anonymization, consent), ethical limitations, explicit
  intended-use boundary.
- **README**: architecture diagram (the 5-box diagram from Section 2 is a good basis),
  one-command setup (`docker compose up` should ideally be close to sufficient).
- **Performance report**: compile model metric, business metric (e.g. estimated retention
  value of correctly flagged at-risk customers — a simple assumption-based estimate is
  fine), p95 inference latency (from Stage 7's Grafana data), throughput, error rate,
  drift metric (from Stage 7's simulation).

**Stop condition**: every item in the Section 3 "required final submission" checklist has
a corresponding file/screenshot/section ready to submit.

---

## 6. SUGGESTED REPO STRUCTURE

```
churn360/
├── .github/workflows/ci.yml
├── dags/                      # Airflow DAG definitions
├── data/                      # DVC-tracked (raw + processed), not committed to git directly
├── docker-compose.yml
├── Dockerfile
├── mlruns/                    # MLflow local tracking (or point to a tracking server)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── src/
│   ├── data/                  # Box 1: load, validate, clean, feature-engineer
│   ├── train/                 # Box 2: training + MLflow logging
│   ├── api/                   # Box 3: FastAPI app
│   └── monitoring/            # Box 4: drift-check scripts/custom metrics
├── tests/
├── docs/
│   ├── model_card.md
│   ├── data_card.md
│   └── responsible_ai_statement.md
├── requirements.txt
├── dvc.yaml / .dvc files
└── README.md
```

---

## 7. KEY DECISIONS LOCKED IN (don't silently change these)

- Dataset: IBM Telco Customer Churn (public).
- Scoring pattern: batch, not real-time streaming.
- Models: Logistic Regression (baseline), Random Forest + XGBoost (candidates).
- No subgroup fairness analysis required for this project specifically (unlike some of the
  other 17 project options) — basic SHAP + model/data cards is sufficient for the
  Responsible AI requirement here.
- Build order is the Section 5 stage order — not the Section 3 listing order.
- AWS is last, demo-only, torn down immediately after.

---

## 8. OPEN QUESTIONS FOR CHETAN (flag these back to him, don't assume)

- Solo or team project? (Affects whether stages can be parallelized.)
- MLflow tracking server: local file-based tracking, or a small hosted/remote tracking
  server? (Local is simpler for a course project unless the brief implies otherwise.)
- Airflow: local/standalone mode vs. Docker-ized Airflow — Docker-ized keeps it consistent
  with the rest of the `docker-compose.yml` setup but has a steeper first-time setup.
- Timeline/deadline, so stages can be budgeted realistically.
