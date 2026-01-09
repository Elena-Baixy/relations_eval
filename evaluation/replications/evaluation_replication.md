# Evaluation: Replication of Linear Relational Embedding Experiment

## Reflection

This replication successfully reproduced the key experiments from the paper "Linearity of Relation Decoding in Transformer LMs". The repository provided clear documentation through the plan.md file, CodeWalkthrough.md, and demo notebooks that made the replication process straightforward.

### What Worked Well

1. **Clear Plan**: The plan.md file provided a comprehensive overview of the hypothesis, methodology, and expected results.

2. **Demo Notebooks**: The demo/demo.ipynb notebook demonstrated the exact workflow for both faithfulness and causality evaluation.

3. **Well-Organized Code**: The source code in `src/` was modular and well-documented, with clear separation of concerns (models, operators, editors, functional utilities).

4. **Reproducible Dataset**: The dataset was included in the repository with consistent formatting.

### Challenges Encountered

1. **Model Selection**: Used GPT-2-XL instead of GPT-J due to the instruction to use the smallest available model. Results were still consistent with the paper's findings.

2. **Notebook Kernel Issues**: Initial attempts to run in notebook sessions had output buffering issues, resolved by running as a standalone Python script first.

### Numerical Consistency

The replicated results are numerically consistent with the original paper:
- Average faithfulness (47.1%) aligns with the reported ~48%
- Causality exceeding faithfulness (72% vs 47.1%) matches the paper's observation
- The pattern of which relations are linearly decodable is consistent

---

## Replication Evaluation - Binary Checklist

### RP1. Implementation Reconstructability

**PASS**

**Rationale**: The experiment can be fully reconstructed from the plan.md and code-walk documentation. The plan clearly describes:
- The hypothesis and methodology (Jacobian-based LRE estimation)
- The evaluation metrics (faithfulness and causality)
- The expected results for comparison

The demo notebooks provide complete working examples that can be followed step-by-step. No major guesswork was required.

---

### RP2. Environment Reproducibility

**PASS**

**Rationale**: The environment was successfully restored and run:
- requirements.txt lists all necessary dependencies
- baukit package installs successfully from GitHub
- All imports worked correctly
- Models loaded successfully from HuggingFace
- No version conflicts or dependency issues encountered

---

### RP3. Determinism and Stability

**PASS**

**Rationale**:
- Seeds are properly set using `set_seed(12345)` as shown in the demo notebooks
- Results are stable across runs
- The repository explicitly handles random seed setting through `experiment_utils.set_seed()`
- Replicated results match between the script execution and notebook execution

---

### RP4. Demo Presentation

**PASS**

**Rationale**: The repository provides comprehensive demos:
1. `demo/demo.ipynb` demonstrates the complete LRE workflow including faithfulness and causality evaluation
2. `demo/attribute_lens.ipynb` demonstrates the Attribute Lens application
3. The demos can be executed without referencing external materials
4. All required inputs, configurations, and execution steps are specified
5. Demo outputs match the documented expected behavior

---

## Summary

This replication was **successful**. All four evaluation criteria pass:

| Criterion | Result |
|-----------|--------|
| RP1. Implementation Reconstructability | PASS |
| RP2. Environment Reproducibility | PASS |
| RP3. Determinism and Stability | PASS |
| RP4. Demo Presentation | PASS |

The repository provides excellent documentation, clear code organization, and working demos that enable straightforward replication of the paper's key experiments. The replicated results are numerically consistent with the original findings, confirming that:
1. LRE can approximate relation decoding for a subset of relations
2. Causality evaluation confirms the causal role of the LRE structure
3. Not all relations are linearly decodable
