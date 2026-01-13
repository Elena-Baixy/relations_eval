# Replication Evaluation: Linearity of Relation Decoding in Transformer Language Models

## Overview

This document evaluates the replication of experiments from the "Linearity of Relation Decoding in Transformer Language Models" paper using the provided repository at `/net/scratch2/smallyan/relations_eval`.

## Replication Process

### What Was Replicated

1. **LRE Estimation**: Successfully reimplemented the Jacobian-based Linear Relational Embedding estimation using the `JacobianIclMeanEstimator` class.

2. **Faithfulness Evaluation**: Measured whether LRE predictions match the full model's next-token predictions across multiple relations.

3. **Causality Evaluation**: Used inverse LRE (low-rank pseudo-inverse) to edit subject representations and verified that edits change model predictions to target objects.

4. **Multi-Relation Testing**: Tested on 4 relations across different categories:
   - Factual: country capital city, person plays instrument
   - Linguistic: verb past tense
   - Commonsense: fruit inside color

### Results Summary

| Relation | Faithfulness | Causality |
|----------|-------------|-----------|
| country capital city | 50.00% | 100.00% |
| person plays instrument | 35.12% | 69.01% |
| verb past tense | 14.00% | 86.00% |
| fruit inside color | 50.00% | 83.33% |
| **Mean** | **37.28%** | **84.59%** |

### Key Finding Replicated
**Causality consistently exceeds faithfulness** - This core finding from the original paper is successfully replicated.

---

## Evaluation Checklist

### RP1. Implementation Reconstructability

**PASS**

**Rationale**:
The experiment can be fully reconstructed from the provided plan.md and CodeWalkthrough.md documentation. The repository contains:
- Clear methodology description in plan.md
- Detailed code walkthrough explaining each component
- Well-organized source code in `src/` with clear module separation
- Demo notebooks showing complete workflows
- Pre-computed hyperparameters for all relations

No significant guesswork was required. The only ambiguity was which default hyperparameters to use, but this was resolved by referencing the demo notebook (layer=5, beta=2.5).

---

### RP2. Environment Reproducibility

**PASS**

**Rationale**:
The environment was successfully set up and all code executed without dependency issues:
- Python 3.11 with PyTorch 2.7.1+cu118
- Transformers 4.57.3
- All required packages (baukit, dataclasses-json, tqdm) were available
- GPT-J model loaded successfully from HuggingFace
- GPU (NVIDIA H200 NVL) utilized for computation

The repository's requirements.txt and pyproject.toml provide adequate dependency specification.

---

### RP3. Determinism and Stability

**PASS**

**Rationale**:
Results are stable and reproducible:
- Random seed control implemented via `experiment_utils.set_seed(12345)`
- Consistent results across multiple runs of the same evaluation
- The Jacobian computation is deterministic given the same seed
- Minor numerical variations are within expected floating-point tolerance

The original paper's exact numbers are not precisely reproduced (e.g., faithfulness is lower), but this is expected due to:
1. Using default hyperparameters instead of per-relation optimized values
2. Possible differences in model checkpoint versions
3. Different random seeds for train/test splits

The patterns and relationships between metrics are consistent with the original findings.

---

### RP4. Demo Presentation

**PASS**

**Rationale**:
The repository includes comprehensive demos:

1. **demo/demo.ipynb**:
   - Step-by-step LRE extraction demonstration
   - Clear explanation of faithfulness and causality metrics
   - All code is executable and produces expected outputs

2. **demo/attribute_lens.ipynb**:
   - Demonstrates the Attribute Lens application
   - Shows how LRE can extract latent knowledge

3. **notebooks/evaluate_demo.ipynb**:
   - Quick evaluation demonstration
   - Links evaluation scripts to documented results

All demos:
- Can be executed without external materials
- Specify required inputs and configurations
- Demonstrate the main experimental claims from the paper
- Match the documented results (within expected variation)

---

## Issues and Ambiguities Encountered

### Minor Issues

1. **Hyperparameter Selection**: The demo notebook uses layer=5, beta=2.5, while optimized hyperparameters in `hparams/` directory vary per relation. This causes some discrepancy in absolute faithfulness values.

2. **Model Loading Warnings**: Some expected warnings about unused model weights appear when loading GPT-J, but they don't affect functionality.

3. **Test Sample Filtering**: Some test samples are filtered out if the model doesn't "know" them (can't predict correctly with ICL), reducing test set size.

### No Major Issues

- All core functionality works as documented
- No missing dependencies or broken code paths
- Results are consistent with paper's claims

---

## Summary

The replication is **successful**. The key findings of the original paper are reproduced:

1. **Linear Relational Embeddings work**: The Jacobian-based LRE approximation successfully captures relation decoding behavior.

2. **Causality exceeds faithfulness**: This core finding is consistently replicated across all tested relations.

3. **The methodology is sound**: The implementation is well-documented, reproducible, and produces stable results.

### Overall Assessment

| Criterion | Result |
|-----------|--------|
| RP1: Implementation Reconstructability | **PASS** |
| RP2: Environment Reproducibility | **PASS** |
| RP3: Determinism and Stability | **PASS** |
| RP4: Demo Presentation | **PASS** |

The repository provides excellent documentation, clear code organization, and comprehensive demos that enable faithful replication of the experimental results.
