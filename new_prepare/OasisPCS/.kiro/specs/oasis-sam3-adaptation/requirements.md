# Requirements Document

## Introduction

Oasis Company is taking over the SAM 3 (Segment Anything with Concepts) project from Meta. This spec covers the documentation overhaul, next update planning, and adaptation of SAM 3 for Oasis's product ecosystem. SAM 3 is a unified foundation model for promptable segmentation in images and videos, supporting text and visual prompts, open-vocabulary concept detection, and multi-object tracking.

The goal is to rebrand, document, extend, and productize SAM 3 under Oasis, while preserving the core model capabilities and improving developer experience, deployment ergonomics, and API usability.

## Glossary

- **SAM3**: Segment Anything Model 3 — the core segmentation model being adapted
- **Oasis_Platform**: The Oasis Company's product platform that will host and distribute SAM 3
- **Developer**: An engineer or researcher integrating SAM 3 into an application
- **Operator**: An Oasis team member managing deployments and model updates
- **Inference_API**: The HTTP/REST API layer wrapping SAM 3 model inference
- **Checkpoint**: A saved model weight file hosted on Hugging Face or Oasis storage
- **SA_Co**: The Segment Anything Concepts benchmark dataset used for evaluation
- **PBT_Runner**: The property-based test runner used for correctness validation
- **Processor**: The Sam3Processor or video predictor interface used for inference

## Requirements

### Requirement 1: Project Documentation Overhaul

**User Story:** As a Developer, I want comprehensive and up-to-date documentation, so that I can quickly understand, install, and use SAM 3 under Oasis.

#### Acceptance Criteria

1. THE Oasis_Platform SHALL provide a top-level README that reflects Oasis branding, contact information, and support channels
2. WHEN a Developer visits the documentation, THE Oasis_Platform SHALL present installation instructions compatible with Python 3.12 and PyTorch 2.7+
3. THE Oasis_Platform SHALL include a CHANGELOG file tracking all changes made by Oasis from the original Meta release
4. WHEN a Developer requests API reference documentation, THE Oasis_Platform SHALL provide auto-generated API docs covering all public classes and functions in the `sam3` package
5. THE Oasis_Platform SHALL include a CONTRIBUTING guide updated with Oasis-specific contribution workflows, code style, and review processes
6. WHEN a Developer encounters an error, THE Oasis_Platform SHALL provide a TROUBLESHOOTING guide covering common installation and inference issues

---

### Requirement 2: Inference API Layer

**User Story:** As a Developer, I want a stable REST API to run SAM 3 inference, so that I can integrate segmentation into applications without managing model code directly.

#### Acceptance Criteria

1. THE Inference_API SHALL expose a `/segment/image` endpoint that accepts an image and a text or visual prompt and returns masks, bounding boxes, and confidence scores
2. THE Inference_API SHALL expose a `/segment/video` endpoint that accepts a video path or stream and a text prompt and returns per-frame segmentation results
3. WHEN a request is received with an invalid or missing prompt, THE Inference_API SHALL return a 400 status code with a descriptive error message
4. WHEN a request is received with an unsupported media type, THE Inference_API SHALL return a 415 status code
5. WHEN the model checkpoint is unavailable or fails to load, THE Inference_API SHALL return a 503 status code and log the failure
6. THE Inference_API SHALL respond to image segmentation requests within 5 seconds for images up to 1920x1080 resolution on a single GPU
7. THE Inference_API SHALL support batch image requests of up to 8 images per call

---

### Requirement 3: Model Checkpoint Management

**User Story:** As an Operator, I want a reliable checkpoint management system, so that I can update, roll back, and distribute model weights without downtime.

#### Acceptance Criteria

1. THE Oasis_Platform SHALL support loading checkpoints from both Hugging Face Hub and a local file path
2. WHEN a checkpoint download fails, THE Oasis_Platform SHALL retry up to 3 times before raising an error
3. THE Oasis_Platform SHALL validate checkpoint integrity using a SHA-256 hash before loading
4. WHEN a new checkpoint is available, THE Oasis_Platform SHALL provide a CLI command to download and verify it
5. IF a checkpoint hash does not match the expected value, THEN THE Oasis_Platform SHALL reject the checkpoint and raise a descriptive error

---

### Requirement 4: Python SDK Improvements

**User Story:** As a Developer, I want a clean and well-typed Python SDK, so that I can integrate SAM 3 into my codebase with confidence.

#### Acceptance Criteria

1. THE Oasis_Platform SHALL provide typed Python interfaces (using `typing` or `dataclasses`) for all public-facing model inputs and outputs
2. WHEN a Developer calls `set_text_prompt` with an empty string, THE Processor SHALL raise a `ValueError` with a descriptive message
3. WHEN a Developer calls `set_image` with a non-image input, THE Processor SHALL raise a `TypeError` with a descriptive message
4. THE Oasis_Platform SHALL publish the SDK to PyPI under the package name `oasis-sam3`
5. THE Oasis_Platform SHALL maintain backward compatibility with the existing `sam3` import namespace for at least one major version

---

### Requirement 5: Evaluation and Benchmarking Pipeline

**User Story:** As an Operator, I want a reproducible evaluation pipeline, so that I can measure model performance on SA-Co benchmarks after any update.

#### Acceptance Criteria

1. THE Oasis_Platform SHALL provide a single CLI command to run the full SA-Co/Gold image evaluation
2. THE Oasis_Platform SHALL provide a single CLI command to run the full SA-Co/VEval video evaluation
3. WHEN an evaluation run completes, THE Oasis_Platform SHALL output a structured JSON report with cgF1, AP, and pHOTA metrics
4. WHEN evaluation results deviate more than 2% from the baseline SAM 3 metrics, THE Oasis_Platform SHALL emit a warning in the evaluation report
5. THE Oasis_Platform SHALL support running evaluations on a configurable subset of the dataset for faster iteration

---

### Requirement 6: Deployment and Packaging

**User Story:** As an Operator, I want containerized deployment artifacts, so that I can deploy SAM 3 inference reliably across environments.

#### Acceptance Criteria

1. THE Oasis_Platform SHALL provide a Dockerfile that builds a runnable SAM 3 inference container with all dependencies
2. WHEN the container starts, THE Oasis_Platform SHALL automatically load the default checkpoint and expose the Inference_API on port 8080
3. THE Oasis_Platform SHALL provide a `docker-compose.yml` for local development that mounts model checkpoints as a volume
4. WHERE GPU support is available, THE Oasis_Platform SHALL configure the container to use CUDA 12.6 or higher
5. THE Oasis_Platform SHALL provide a health check endpoint at `/health` that returns 200 when the model is loaded and ready

---

### Requirement 7: Next Model Update — SAM 3.2 Planning

**User Story:** As an Operator, I want a planned roadmap for SAM 3.2, so that Oasis can prioritize improvements that close the gap between model and human performance.

#### Acceptance Criteria

1. THE Oasis_Platform SHALL document target improvements for SAM 3.2 including increased cgF1 on SA-Co/Gold from 54.1 toward 65.0
2. THE Oasis_Platform SHALL document a plan to improve video cgF1 on SA-V test from 30.3 toward 45.0
3. THE Oasis_Platform SHALL include a data augmentation strategy targeting rare and fine-grained concepts underrepresented in the current training set
4. THE Oasis_Platform SHALL document an architecture experiment plan for improving the presence token mechanism to better handle ambiguous or overlapping concepts
5. WHEN a SAM 3.2 experiment is completed, THE Oasis_Platform SHALL record results in a structured experiment log with model config, dataset, and metric deltas
