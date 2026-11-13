# AI-Assisted Support Triage

An AI-assisted support triage pipeline that retrieves policy evidence, prioritises incoming tickets, drafts customer responses, proposes action plans, and validates whether generated claims are grounded in the available evidence.

The project focuses on traceability and conservative handling: unsupported claims should be rejected or revised rather than presented as established facts.

Automated outputs are preliminary recommendations. They do not replace human review or authorise irreversible customer-account actions.

## Features

- **Document ingestion:** Load and chunk knowledge-base documents while preserving document-level metadata.
- **Evidence retrieval:** Retrieve relevant policy and procedural evidence, with policy-aware retrieval and source references.
- **Ticket triage:** Categorise tickets, assess confidence, and rank the support queue.
- **Response drafting:** Generate customer-facing response drafts using retrieved evidence.
- **Action planning:** Produce suggested next steps for support agents.
- **Grounding validation:** Identify unsupported claims and reconcile rejected claims.
- **NLI verification:** Optionally use a Natural Language Inference model for semantic evidence verification.
- **Auditability:** Preserve LLM-call metadata, pipeline-stage records, source references, and generated artifacts.
- **Evaluation:** Run response-quality, relevance, regression, and validation checks.

## Pipeline

The pipeline in `src/main.py` covers these stages:

1. Load tickets and knowledge-base documents.
2. Chunk and index knowledge documents with document-level metadata.
3. Normalise incoming tickets.
4. Retrieve relevant evidence.
5. Triage tickets, assess confidence, and rank the support queue.
6. Generate customer response drafts.
7. Generate action plans.
8. Validate response grounding and identify unsupported claims.
9. Reconcile rejected claims and produce final ticket outputs.
10. Persist output artifacts, audit records, and pipeline-stage results.

The LLM workflow separates triage, response drafting, action planning, and grounding validation. Retrieval evidence and source references are preserved through the pipeline.

When evidence is insufficient, the system uses conservative handling and fallback behaviour. Cases requiring additional judgement should be referred for human review.

## Repository structure

- `src/` — pipeline implementation, retrieval, triage, drafting, grounding, evaluation, and review modules.
- `knowledge_base/` — policy and support documents used as evidence.
- `tickets.json` — sample input tickets.
- `evaluation/` — response-quality datasets and expected labels used by evaluation scripts and regression tests.
- `test_*.py` — automated unit, integration, and regression tests.
- `README_BLIND_BENCHMARK.md` — blind evaluation notes.
- `README_FRESH_BENCHMARK.md` — fresh evaluation procedure.
- `assumptions.md` — documented implementation assumptions.
- `requirements.txt` — lightweight core dependencies.
- `requirements-test.txt` — dependencies for the test profile.
- `requirements-live.txt` — additional dependencies for live model execution.

The `output/` and `logs/` directories are created at runtime. They are not required as precomputed inputs: the pipeline generates its artifacts when executed.

## Requirements

- Python 3.12
- Linux or WSL recommended for the documented commands
- Internet access for installing dependencies and downloading live models
- API credentials for live LLM provider execution

Three dependency profiles are provided:

| Profile | Purpose |
|---|---|
| Core | Lightweight offline/mock execution |
| Test | Automated test suite |
| Live CPU | Real embedding and NLI inference, plus live LLM providers |

The profiles were verified in fresh virtual environments. Core offline execution passed, the test profile passed all 280 tests, and the live CPU profile passed real embedding and NLI inference plus all 280 tests.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/LoboVH/ai-assisted-support-triage.git
cd ai-assisted-support-triage

```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install core dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the pipeline in mock mode

```bash
LLM_MODE=mock NLI_MODE=mock python src/main.py
```

No API key is required for mock execution. The pipeline reads `tickets.json` and the documents under `knowledge_base/`, then generates runtime artifacts under `output/`.

## Running the tests

Install the test profile:

```bash
python -m pip install -r requirements-test.txt
```

Run the complete test suite:

```bash
python -m unittest discover -v
```

The test profile avoids installing the large PyTorch and Sentence Transformers stack. Tests that exercise optional live integrations use mocks.

## Live model execution

### CPU-only installation

For CPU-only execution on Linux or WSL:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-live.txt
```

### Configure an LLM provider

Create a local environment file:

```bash
cp -n .env.example .env
```

Edit `.env` and provide the API key for the selected provider.

**Groq**

```dotenv
LLM_MODE=live
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL_NAME=openai/gpt-oss-20b
```

**Google Gemini**

```dotenv
LLM_MODE=live
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

The Groq model can be changed through `GROQ_MODEL_NAME`.

The client supports provider fallback when another configured provider is available. For an isolated provider test, leave the other provider's API key unset.

Keep `.env` local and uncommitted. Never commit API keys or place credentials in source files, shell history, logs, or documentation.

### Run with a live LLM provider

```bash
LLM_MODE=live python src/main.py
```

Live requests may encounter quota limits or temporary provider errors. The client retries selected provider errors and may fall back to another configured provider.

## NLI configuration

NLI (Natural Language Inference) verification can be configured independently of LLM generation.

- By default, NLI follows `LLM_MODE`: mock/off mode uses offline NLI, while live mode attempts to load the semantic NLI model.
- `NLI_MODE=live` enables the CrossEncoder NLI model independently of LLM generation mode.
- `NLI_MODE=mock` forces offline NLI verification.
- `SUPPORT_TRIAGE_OFFLINE=1` takes precedence and forces offline operation.

For mock LLM generation with real semantic NLI verification:

```bash
LLM_MODE=mock NLI_MODE=live python src/main.py
```

**Important:** `NLI_MODE=live` controls NLI verification only. It does not enable live LLM API calls. Set `LLM_MODE=live` separately when using an external provider.

Offline NLI cannot independently verify substantive claims; it fails closed rather than treating missing model predictions as evidence of entailment.

## Validate generated artifacts

After running the pipeline:

```bash
PYTHONPATH=src python -m validate
```

The validator checks generated artifacts, ticket coverage, knowledge chunks, pipeline stages, and audit records. It does not execute the pipeline itself.

To validate revised response drafts:

```bash
PYTHONPATH=src python -m validate_revision
```

## Evaluate response quality

Evaluate the original response drafts:

```bash
python -m src.evaluate_response_quality
```

Evaluate a selected draft file:

```bash
python -m src.evaluate_response_quality --drafts output/revised_response_drafts.json
```

The default evaluator writes to `output/response_quality_evaluation.json`. Running it again overwrites that report.

Additional evaluation scripts:

```bash
python src/evaluate_response_quality_fresh_fixed.py
python src/evaluate_response_quality_challenge.py
```

These scripts write timestamped reports under `output/`.

### Interpreting evaluation results

At the verified development checkpoint:

- Fresh diagnostic evaluation: 40/40 full-case agreement with curated expected labels.
- Challenge evaluation: 32/32 agreement after evaluator improvements.
- The initial challenge evaluation was 19/32 (59.38%), exposing missed critical failures.
- The evaluator was subsequently improved with contrast-aware negation handling and additional unsupported-claim phrase coverage.

These results measure agreement with internally curated labels. They are not independently adjudicated third-party benchmark results, nor proof of real-world or end-to-end pipeline accuracy.

The fresh and challenge evaluators share the underlying `evaluate_ticket()` implementation. Their results should not be presented as independent validations of the evaluation logic.

Avoid tuning the evaluator against the same challenge set and then describing its result as independent validation.

## Human review

The project includes tools for recording human review of generated response drafts.

Review the original drafts:

```bash
python -m src.record_human_review
```

Review revised drafts:

```bash
python -m src.record_human_review --drafts output/revised_response_drafts.json
```

Review records are stored locally in `evaluation/human_reviews.json`. This file is generated when reviews are recorded and is intentionally not part of the published sample data.

Reviews are matched to the exact draft text using a hash. Approval of an earlier draft should not be treated as approval of a subsequently revised draft.

## Generated artifacts and audit records

Depending on the commands executed, `output/` may contain:

- `knowledge_corpus.json`
- `retrieval_results.json`
- `triage.json` and `triage_results.json`
- `response_drafts.json`
- `action_plan.json` and `action_plans.json`
- `grounding_results.json` and `grounding_validation.json`
- `final_ticket_outputs.json`
- `queue_ranking.json`
- Response-quality and relevance evaluation reports
- Additional validation and audit artifacts

The `logs/` directory contains pipeline-stage and LLM-call logs.

The LLM audit log records information about model calls, including provider, model, stage, prompt metadata, ticket references, and artifact relationships. Cache hits and actual provider requests should be distinguished when interpreting audit counts.

Running the pipeline or evaluators may create or replace files under `output/`. Preserve any run artifacts you need before executing another run.

## Verified development results

The dependency profiles were tested in fresh virtual environments:

| Profile | Verification |
|---|---|
| Core | Dependency check passed; offline pipeline execution passed |
| Test | Dependency check passed; 280 tests passed |
| Live CPU | Dependency check passed; real embedding and NLI inference passed; 280 tests passed |

The live CPU environment used CPU-only PyTorch and did not require CUDA.

A representative live development run processed four tickets and completed the pipeline through `RESULTS_FINALISED`. Its audit contained 16 records, including cache hits, and 12 actual Groq requests. Strict artifact validation passed for that run.

These results describe specific development environments and runs. They do not guarantee that a fresh environment, different input dataset, provider, or model will produce identical results.

## Limitations

- Evaluation datasets use internally curated expected labels rather than independent third-party adjudication.
- Phrase-based claim detection can miss unfamiliar paraphrases and complex natural-language constructions.
- Live provider behaviour depends on API availability, quotas, credentials, and model responses.
- Semantic NLI verification depends on model availability and its predictions.
- Automated grounding and quality checks do not eliminate the need for human review.
- Generated response drafts and action plans are recommendations, not authorisation to perform irreversible actions.

## Reproducible execution

For a fresh run:

1. Install the required dependency profile.
2. Configure the LLM and NLI modes.
3. Provide input tickets in `tickets.json` and knowledge documents under `knowledge_base/`.
4. Execute `python src/main.py`.
5. Validate the generated artifacts.
6. Run the offline test suite.

The pipeline reads its inputs from disk and regenerates its outputs. It should not depend on precomputed output artifacts being present.