# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedSigLIP is a medical variant of SigLIP (Sigmoid Loss for Language Image Pre-training) — a 400M vision encoder + 400M text encoder that maps medical images and text into a shared embedding space. It supports 448x448 image resolution with up to 64 text tokens. Designed for data-efficient classification, zero-shot classification, and semantic image retrieval (not text generation).

## Repository Structure

- `python/serving/` — Vertex AI serving container: Gunicorn HTTP server + Triton Inference Server backend
  - `predictor.py` — Core prediction logic: parses requests, preprocesses images/text via SiglipImageProcessor/SiglipTokenizer, batches inputs, calls model, assembles responses
  - `server_gunicorn.py` — Entry point; launches Gunicorn with `MedSiglipPredictor` via the serving framework
  - `async_batch_predictor.py` — Thread-pool-based parallel/batch prediction orchestration
  - `flags.py` — All configurable flags (absl-py), with env var defaults
  - `serving_framework/` — Reusable Vertex AI serving library: Flask+Gunicorn HTTP server, `PredictionExecutor` abstraction, `ModelRunner` interface, Triton integration
  - `model_repository/default/1/model.py` — Triton PyTorch backend model wrapper (`SiglipWrapper`); loads weights from HF or Vertex GCS
  - `Dockerfile` — Container image based on `nvcr.io/nvidia/tritonserver`, two-stage Python env (Triton's Python + venv for server)
  - `entrypoint.sh` — Launches Triton server + Gunicorn frontend in parallel
- `python/data_accessors/` — Pluggable data source layer with a common `AbstractDataAccessor` interface
  - Implementations: `inline_bytes/`, `inline_text/`, `http_image/`, `gcs_generic/`, `dicom_generic/`, `dicom_wsi/`
  - `local_file_handlers/` — File format handlers: traditional images, DICOM, OpenSlide, WSI DICOM
  - Each data accessor has: `data_accessor.py`, `data_accessor_definition.py` (JSON parsing), `data_accessor_test.py`
- `python/data_processing/` — Image encoding utilities (PNG encode/decode)
- `python/pre_processor_configs/` — SigLIP tokenizer configs (downloaded from HuggingFace)
- `notebooks/` — Jupyter notebooks for HuggingFace and Vertex AI workflows (quick start, fine-tuning, classification)

## Architecture

**Request flow:** HTTP request → Gunicorn (`server_gunicorn.py`) → `PredictionExecutor.execute()` → `MedSiglipPredictor.predict()` → parse instances via data accessors → load image/text data (parallel via ThreadPool) → preprocess (SiglipImageProcessor/SiglipTokenizer) → batch and call model via `ModelRunner` (Triton gRPC) → assemble embedding response JSON.

**Key abstractions:**
- `AbstractDataAccessor[InstanceDataClass, InstanceDataType]` — generic interface for all input sources; has `load_data()`, `data_iterator()`, `__len__()`
- `ModelRunner` — abstract model execution interface; `TritonServerModelRunner` is the concrete implementation communicating via gRPC
- `PredictionExecutor` — abstract request handler; `InlinePredictionExecutor` bridges predictor functions to the server
- `AsyncBatchModelPredictor` — manages parallel data loading and batched model inference

## Running Tests

Tests use `absl.testing.absltest` (not pytest). Run from the `python/` directory:

```bash
# Run a single test file
cd python && python -m <module_path_to_test>
# Example:
cd python && python -m serving.predictor_test
cd python && python -m data_accessors.inline_bytes.data_accessor_test
cd python && python -m serving.serving_framework.triton.triton_server_model_runner_test
```

## Dependencies

Dependencies are managed with `pip-compile` and pinned with hashes in lock files:
- `python/serving/requirements.in` — top-level serving deps (pulls in `data_accessors/requirements.in`, `serving_framework/requirements.in`, `serving_framework/triton/requirements.in`)
- `python/serving/requirements.txt` — compiled/locked version
- Key libraries: `transformers`, `absl-py`, `flask`, `gunicorn`, `grpcio`, `numpy`, `pillow`, `pydicom`, `ez-wsi-dicomweb`, `opencv-python-headless`, `openslide-python`, `redis`

## Conventions

- Apache 2.0 license header on all source files
- Python imports use package-relative paths from `python/` as the root (e.g., `from data_accessors import ...`, `from serving import ...`)
- Configuration flags use `absl.flags` with environment variable fallbacks
- Test files are co-located with source: `<module>_test.py` next to `<module>.py`
- Data accessor pattern: each source type has a directory with `data_accessor.py` (implementation), `data_accessor_definition.py` (request JSON → dataclass parsing), and `data_accessor_test.py`
