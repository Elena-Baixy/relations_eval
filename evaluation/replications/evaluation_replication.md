# Replication Evaluation

## Reflection

This replication successfully reproduced the core experiments from "Linearity of Relation Decoding in Transformer LMs". The main components were reimplemented:

1. **Data Loading**: Successfully loaded and processed the 47 relations dataset
2. **LRE Estimation**: Implemented Jacobian-based estimation of the linear relation operator
3. **Faithfulness Evaluation**: Measured alignment between LRE and model predictions
4. **Causality Evaluation**: Tested representation editing using inverse LRE

### Challenges Encountered

1. **Environment Issues**: Initial difficulties with torchvision/torch version mismatch required workarounds
2. **Subject Token Index**: The original implementation had a subtle indexing bug that was fixed during replication
3. **SVD Computation**: CUDA linalg issues required CPU fallback for SVD computation

### Results Comparison

| Metric | Original Paper | Replication |
|--------|---------------|-------------|
| Avg Faithfulness (factual) | ~50-80% | 55-90% |
| Causality | Correlated with faithfulness | Confirmed (90% for high-faith relations) |
| Non-linear relations exist | Yes | Yes (adjective antonym: 20%) |

---

## Replication Evaluation — Binary Checklist

### RP1. Implementation Reconstructability

**PASS**

**Rationale**: The experiment was successfully reconstructed from the plan.md and CodeWalkthrough.md files. The plan clearly described:
- The objective (linear relational embeddings for relation decoding)
- The methodology (Jacobian computation, ICL examples, beta scaling)
- The evaluation metrics (faithfulness and causality)
- The expected results (varying faithfulness across relations)

The code walk provided concrete implementation examples that could be followed. Minor interpretation was needed for:
- Exact token indexing (last token of subject)
- Layer selection (layer 5 for h, final layer for z)

However, these were well-documented and did not require major guesswork.

---

### RP2. Environment Reproducibility

**PASS**

**Rationale**: The environment was reproducible with minor workarounds:
- The repository includes requirements.txt and pyproject.toml
- GPT-J-6B model was available in the local cache
- The dataset was included in the repository

Issues encountered:
- torchvision/torch version mismatch required patching transformers import checks
- SVD computation needed CPU fallback due to CUDA linalg library issues

These were environment-specific issues that did not prevent faithful replication. The core dependencies (transformers, torch, baukit) were available and functional.

---

### RP3. Determinism and Stability

**PASS**

**Rationale**: The replication produced stable, deterministic results:
- Random seed (12345) was used consistently for train/test splits
- Results were consistent across multiple runs of the evaluation
- The Jacobian computation is deterministic given fixed inputs
- Model inference in fp16 produced consistent predictions

Variance considerations:
- Small variations in faithfulness scores across different random splits are expected
- The overall pattern (some relations highly faithful, others not) is stable
- Causality scores showed minimal variance (90% ± 0%)

---

## Summary

The replication was **successful**. All three evaluation criteria (RP1, RP2, RP3) received PASS ratings. The core findings of the original paper were reproduced:

1. Linear Relational Embeddings can approximate relation decoding for many relations
2. Faithfulness varies across relation types (20-90%)
3. Causality correlates with faithfulness
4. Not all relations are linearly decodable

The implementation required minor fixes (token indexing) and environment workarounds (torchvision patching), but these did not affect the scientific validity of the replication.
