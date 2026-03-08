# Repository Audit — MedSigLIP

> **Date:** 2026-03-08
> **Scope:** Full repository audit covering serving code, experiment infrastructure, data pipeline, and deployment

---

## 1. Repository Overview

| Component | Files | Tests | Coverage |
|-----------|-------|-------|----------|
| Serving (predictor, batch, server) | 12 | 5 | Good for unit; no integration tests |
| Serving Framework (Flask, Gunicorn, Triton) | 8 | 4 | Good |
| Data Accessors (6 types + handlers + utils) | 30+ | 15+ | Good |
| Data Processing | 2 | 1 | Adequate |
| Pre-processor Configs | 2 | 1 | Adequate |
| Experiments (data, models, training, utils) | 15 | 0 | **No tests** |
| Notebooks | 5 | — | N/A |

---

## 2. Critical Issues

### HIGH — Production Risks

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| H1 | **Thread pool timeout not caught** | `async_batch_predictor.py:173` | `concurrent.futures.TimeoutError` crashes the endpoint. Only `DataAccessorError` is caught. |
| H2 | **Gunicorn timeout < thread pool timeout** | `server_gunicorn.py:49` vs `flags.py` | Gunicorn kills workers at 120s, but thread pool allows 1800s. Orphaned requests, wasted compute. |
| H3 | **HTTP requests without timeouts** | `http_image/data_accessor.py:41`, `server_health_check.py:22` | `requests.get()` can hang indefinitely on DNS/connection issues. |
| H4 | **Prediction validation disabled** | `server_gunicorn.py:63` | `prediction_validator = None` with TODO comment. Invalid responses reach clients. |
| H5 | **No CI/CD pipeline** | `.github/` | Issue/PR templates exist but no GitHub Actions. Tests only run manually. |

### MEDIUM — Code Quality

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| M1 | **Exception chains broken** | Throughout data accessors | `DataAccessorError` wraps exceptions without `from e`, losing original tracebacks |
| M2 | **Global mutable singleton** | `predictor.py:85-145` | Image processor/tokenizer use module-level globals with lazy init. Thread-safe via GIL but fragile |
| M3 | **Assertions used for validation** | `triton_server_model_runner.py:94` | `assert result is not None` — asserts can be stripped with `python -O` |
| M4 | **JSON newline stripping** | `serving_framework/server_gunicorn.py:94` | `json.dumps().replace("\n", "")` — breaks if JSON string values contain literal `\n` |
| M5 | **Windowing function has known bugs** | `data_processing/image_utils.py:128-150` | Comments document bugs kept "for backward compatibility" |
| M6 | **No experiment tests** | `experiments/` | No unit tests for dataset, caption builder, models, or training logic |

### LOW — Minor Issues

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| L1 | `MAX_EMBEDDINGS_PER_REQUEST` uses -1 magic value | `flags.py:49` | Implicit; should be explicit constant or None |
| L2 | Hard-coded port 8500 in entrypoint | `entrypoint.sh:35` | No env var override |
| L3 | Dockerfile clones full repos for licenses | `Dockerfile:53-73` | `.git` dirs add image size |
| L4 | No Dockerfile HEALTHCHECK instruction | `Dockerfile` | Container won't self-report failures to orchestrator |
| L5 | Race condition: Triton + Gunicorn start concurrently | `entrypoint.sh:41-50` | Gunicorn may connect before Triton is ready (mitigated by health check in server code) |

---

## 3. Experiment Infrastructure Audit

### Strengths
- Clean modular architecture: `data/`, `models/`, `training/`, `utils/`
- Comprehensive caption builder with tokenization-aware truncation
- Clinical evaluation metrics (sensitivity @ specificity) — directly relevant for deployment
- Stratified splits on composite key — handles multi-task label correlation
- Shell scripts for batch orchestration
- W&B integration for experiment tracking
- Config snapshots saved per experiment for reproducibility

### Weaknesses

| Issue | Impact | Recommendation |
|-------|--------|---------------|
| **No unit tests** for any experiment code | Regressions go undetected | Add tests for `caption_builder`, `dataset`, `splits`, `metrics` |
| **No pretrained baseline evaluation** | Can't measure fine-tuning improvement | Run zero-shot eval with pretrained model before fine-tuning |
| **No test set evaluation pipeline** | Phase 2 results not on held-out test | Implement final evaluation on 628 test images |
| **No checkpoint management strategy** | Checkpoints accumulate without cleanup | Add checkpoint rotation (keep best + last only) |
| **No reproducibility verification** | Same seed doesn't guarantee identical runs across hardware | Run one experiment twice to verify |
| **No error recovery** | If training crashes, must restart from scratch | Enable `resume_from_checkpoint` in Trainer config |
| **MultiTaskClassifier defined but unused** | Dead code | Either integrate into Phase 2 or remove |
| **No learning rate finder** | LR choices are manual | Consider LR range test before committing to full runs |

---

## 4. Data Pipeline Audit

### Strengths
- Thorough corrupt image detection (15,131 files identified and filtered)
- Stratified splits preserving label distribution
- Test set properly excluded from train/val
- Class weight clamping prevents extreme gradient updates
- Caption truncation respects tokenizer limits

### Weaknesses

| Issue | Impact | Recommendation |
|-------|--------|---------------|
| Image resize uses `Resize(448)` only | Doesn't crop; aspect ratio may differ from MedSigLIP's expected square input | Verify that `SiglipImageProcessor` handles padding/cropping, or add `CenterCrop(448)` |
| No data augmentation | May limit generalization | Consider horizontal flip, color jitter, random crop for classification (not contrastive) |
| Bad images listed but not all removed from disk | Fallback to LOAD_TRUNCATED_IMAGES introduces zero-filled pixels | Complete the move to `visilant_data_bad/` |
| No label noise estimation | Some labels may be incorrect | Inspect high-loss samples after training; consider cleanlab |
| Test set is very small (628) | Bootstrap CI will be wide for rare classes | Acceptable for now; note limitations |

---

## 5. Serving Code Audit

### Architecture: Sound
The layered design (HTTP → PredictionExecutor → Predictor → ModelRunner) is clean and extensible. Data accessor pattern is well-applied across 6 input types.

### Performance Concerns
- `parallel_predict_embeddings()` runs model per-image (not batched) — only used when batch mode fails
- No request queuing or rate limiting
- ThreadPool default of 4 workers may be low for concurrent requests

### Security
- GCS/DICOM store authentication passes through; no credential validation beyond type check
- HTTP image accessor fetches from arbitrary URLs — potential SSRF
- No input size limits beyond `MAX_EMBEDDINGS_PER_REQUEST`

---

## 6. Dependency Audit

### Serving Dependencies
- **transformers:** Actively maintained, no known CVEs in current version
- **torch:** Actively maintained
- **grpcio:** Multiple historical CVEs; ensure latest version
- **flask/gunicorn:** Stable; ensure no deprecated features used
- **pillow:** Actively patched; multiple dependabot PRs visible in branches
- **requests:** Used without timeouts (see H3)

### Experiment Dependencies
- **torch 2.10.0 (CUDA 12.8):** Latest; matches A100 capability
- **transformers 4.51+:** Latest; required for MedSigLIP support
- **peft 0.15.0:** LoRA implementation; stable
- **wandb:** Experiment tracking; no issues
- **uv:** Package manager; reliable and fast

---

## 7. Recommendations Summary

### Immediate (Before More Experiments)

1. **Run pretrained baseline evaluation** — You need this to measure improvement
2. **Verify bad images are fully removed** from visilant_data/ (check move status)
3. **Enable `resume_from_checkpoint`** in training args — prevents data loss on crashes

### Short-Term (During Experiment Phase)

4. Add unit tests for `caption_builder.py`, `dataset.py`, and `metrics.py`
5. Add request timeouts to HTTP image accessor
6. Implement test set evaluation for Phase 2

### Medium-Term (Before Deployment)

7. Fix thread pool timeout catching (H1)
8. Align Gunicorn and thread pool timeouts (H2)
9. Re-enable prediction validation (H4)
10. Add CI/CD pipeline with automated tests
11. Add Dockerfile HEALTHCHECK
12. Fix exception chains in data accessors (add `from e`)
