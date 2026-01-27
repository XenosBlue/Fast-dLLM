# Fast-dLLM v2 — Compute Skipping Extensions (Evaluation Only)

This repository is a public fork of **Fast-dLLM v2** that adds and evaluates **compute-skipping policies at inference time**, without modifying the original training procedure.

The original Fast-dLLM v2 project and full documentation are available here:  
https://github.com/NVlabs/Fast-dLLM/tree/main/v2

This README focuses **only on evaluation**, and documents the added methods, files, and entry points.

---

## Goal

On top of the Fast-dLLM v2 codebase, implement and evaluate two cosine-similarity–based compute-skipping policies:

1. **Token-level skipping across denoising steps**
2. **Layer-level skipping within a denoising step**

Both methods aim to reduce inference FLOPs while preserving accuracy, and are evaluated using the same **LM-Eval** setup as the original Fast-dLLM v2 benchmarks.

---

## Installation

Follow **exactly the same installation steps as the original Fast-dLLM v2 repository**.

Please refer to the original README for environment setup, dependencies, and installation instructions:  
https://github.com/NVlabs/Fast-dLLM/tree/main/v2

No additional dependencies are required for the compute-skipping extensions.

---

## Compute-Skipping Methods

### 1. Token-Level Skipping (Across Denoising Steps)

**Where it operates**
- Across *adjacent denoising steps* within the same block

**Method**
- For each token position, compute the cosine similarity between:
  - The token hidden state at the current denoising step
  - The token hidden state at the previous denoising step
- If the cosine similarity exceeds a threshold (`cosine_threshold`):
  - Skip recomputation of that token
  - Reuse the previous step’s output

**Effect**
- Reduces token-level computation during denoising
- Avoids redundant updates once token representations stabilize

---

### 2. Layer-Level Skipping (Within a Denoising Step)

**Where it operates**
- Between *adjacent transformer layers* inside a single denoising step

**Method**
- For each layer, compute the cosine similarity between:
  - The input hidden states to the current layer
  - The input hidden states to the previous layer
- If the similarity exceeds a threshold (`cosine_threshold`):
  - Skip executing the current transformer layer
  - Propagate the hidden states forward unchanged

**Effect**
- Reduces depth-wise computation
- Conditionally skips layers based on representation stability

---

## Added Files

The following files are added relative to the original Fast-dLLM v2 repository:

### Token-Level Skipping
- `token_skip_generation_functions.py`  
  Implements token-level cosine similarity checks and token reuse during denoising.
- `token_skip_eval.py`  
  LM-Eval evaluation harness using token-level compute skipping.

### Layer-Level Skipping
- `layer_skip_generation_functions.py`  
  Implements cosine-based layer skipping via transformer layer wrappers.
- `layer_skip_eval.py`  
  LM-Eval evaluation harness using layer-level compute skipping.

Each evaluation file serves as an **independent entry point** and replaces `eval.py` for that specific policy.

---

## Evaluation

Evaluation follows the same procedure as Fast-dLLM v2 and uses **LM-Eval**.

Supported tasks include:
- GSM8K
- Minerva-Math
- IFEval
- MMLU
- GPQA

### Entry Points

Instead of `eval.py`, use the policy-specific evaluation scripts:

- **Token-level skipping**
  ```bash
  accelerate launch token_skip_eval.py ...
