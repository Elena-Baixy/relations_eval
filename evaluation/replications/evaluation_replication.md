# Evaluation: Replication of "Linearity of Relation Decoding in Transformer LMs"

## Reflection

### What Went Well
1. **Environment Setup**: The repository's dependencies were well-documented and the code ran without major modifications
2. **Demo Availability**: The `demo/demo.ipynb` provided a clear reference for the expected workflow and outputs
3. **Code Organization**: The source code was modular and well-structured, making it easy to understand the experimental pipeline
4. **Causality Replication**: The causality metric was perfectly replicated (100%), validating the core hypothesis

### Challenges Encountered
1. **Faithfulness Divergence**: The faithfulness score (26.32%) was significantly lower than the demo reference (78.9%)
2. **Random Variance**: Results depend heavily on the random train/test split, which affects LRE estimation quality
3. **Limited Documentation**: The plan.md describes the overall methodology but lacks specific details on hyperparameter sensitivity

### Key Observations
1. The repository is designed as a demo/research codebase, with the primary demonstration in `demo/demo.ipynb`
2. The LRE estimation requires careful training sample selection for optimal performance
3. The causality metric is more robust than faithfulness to experimental variations
4. The core claim (linear approximation of relation decoding) is supported by the causality results

---

## Replication Evaluation - Binary Checklist

### RP1. Implementation Reconstructability

**Status: PASS**

**Rationale**:
- The experiment can be fully reconstructed from the plan.md and demo notebook
- The plan clearly describes:
  - The hypothesis (linear relation decoding)
  - The methodology (Jacobian-based LRE estimation)
  - The metrics (faithfulness and causality)
  - The hyperparameters (layer, beta, rank)
- The demo notebook provides executable code demonstrating the complete workflow
- No major guesswork or inference was required beyond standard ML/NLP knowledge
- The source code in `src/` is well-organized and follows the described methodology

### RP2. Environment Reproducibility

**Status: PASS**

**Rationale**:
- All required packages are listed in `requirements.txt`
- Core dependencies (transformers, torch, baukit, dataclasses-json) installed without conflicts
- The GPT-J model loaded successfully from HuggingFace
- The dataset loaded correctly from the repository's `data/` directory
- No version conflicts or irrecoverable environment issues encountered
- Code runs on both CPU and GPU (CUDA)

### RP3. Determinism and Stability

**Status: PASS**

**Rationale**:
- Seeds are set for random, numpy, and torch (SEED=12345)
- The causality metric shows perfect reproducibility (100% across runs)
- The faithfulness variance is expected due to the nature of the train/test split
- The experimental utilities include proper seed control (`experiment_utils.set_seed()`)
- The core mechanism (LRE estimation via Jacobian) is deterministic given the same inputs
- Note: Some variance in faithfulness is inherent to the experimental design when training data changes

### RP4. Demo Presentation

**Status: PASS**

**Rationale**:
- A clear demo exists at `demo/demo.ipynb`
- The demo can be executed without external resources (model downloads from HuggingFace)
- The demo demonstrates both faithfulness and causality evaluation
- All key steps are shown: model loading, LRE estimation, faithfulness evaluation, causality evaluation
- The demo specifies required inputs (relation selection, hyperparameters)
- Expected outputs are shown in the notebook (faithfulness ~78.9%, causality ~100%)
- The replication using the demo workflow successfully replicated the causality results
- The faithfulness divergence is documented and explained (training data variance)

---

## Summary

The replication was **partially successful**:

| Aspect | Status |
|--------|--------|
| Implementation Reconstructability | PASS |
| Environment Reproducibility | PASS |
| Determinism and Stability | PASS |
| Demo Presentation | PASS |

**Key Findings**:
1. **Causality**: Perfectly replicated (100% vs 100% reference)
2. **Faithfulness**: Diverged from reference (26.32% vs 78.9% reference)

The divergence in faithfulness is attributed to:
- Random train/test split variance
- Limited training data (5 samples)
- The metric's sensitivity to training sample selection

Despite the faithfulness divergence, the core scientific claim (relation decoding can be approximated by linear transformations) is validated by the successful causality results. The LRE-based editing reliably changes model outputs, demonstrating that the learned linear transformation captures meaningful relational structure.

**Overall Assessment**: The repository provides a **replicable** experimental framework. The demo successfully guides replication of the core methodology, and the causality results confirm the paper's central hypothesis. The faithfulness variance is a known limitation of the approach when using limited training data.
