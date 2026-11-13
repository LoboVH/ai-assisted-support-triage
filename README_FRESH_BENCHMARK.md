# Fresh response-quality benchmark pack

Contains 40 newly authored cases for the existing 4-ticket support-triage evaluator, plus a runner.

## Install in the project root

Copy the ZIP to your WSL project root, then run:

```bash
unzip -n support_triage_fresh_benchmark.zip
python -m py_compile src/evaluate_response_quality_fresh.py
python src/evaluate_response_quality_fresh.py
```

The runner makes a temporary copy of `src/evaluate_response_quality_generalization.py`, changes only its input and report paths to point to the fresh dataset and a timestamped report, then runs it. The committed evaluator, existing 30-case regression dataset, and existing probe report are not modified. The temporary evaluator file is deleted after the run.

Reports are created under `output/response_quality_fresh_evaluation_<timestamp>.json` with a companion `.txt` console log.

## Benchmark protocol

- Treat this as the first-pass, held-back evaluation. Do not change scorer rules or these expected labels before reviewing the results.
- Review mismatches manually against the policy source of truth. An exact match against these authored labels is not evidence of real-world accuracy by itself.
- These cases are authored from the four ticket/policy scenarios already established in the project context. They are not independently adjudicated by an external reviewer, so consider asking a domain/policy owner to audit the labels before using the scores in a report.
- After review, preserve this set as another regression suite and make a separate fresh set for any later independent measurement.
