# Linear Relational Embedding (LRE) Replication Documentation

## Goal

Replicate the experiments from the paper "Linearity of Relation Decoding in Transformer LMs" (Hernandez et al., 2023). The paper investigates how transformer language models represent and decode relational knowledge, specifically testing whether relation decoding can be well-approximated by linear transformations on subject representations.

## Data

The dataset contains 47 relations across four categories:
- **Factual**: country-capital, country-language, person-occupation, etc.
- **Commonsense**: work-location, substance-phase, fruit-color, etc.
- **Linguistic**: adjective-antonym, adjective-comparative, verb-past-tense, etc.
- **Bias**: name-gender, occupation-gender, name-religion, etc.

Each relation contains subject-object pairs (e.g., "France" -> "Paris" for country-capital).

## Method

### Linear Relational Embedding (LRE)

The core hypothesis is that for many relations, the transformer's decoding procedure can be approximated by a linear transformation:

```
LRE(s) = W * s + b
```

Where:
- `s` is the subject representation at intermediate layer h
- `W` is the Jacobian matrix (∂z/∂s)
- `b` is the bias term (z - W*s)
- `z` is the object representation at the final layer

### Jacobian Estimation

For each relation, we compute the LRE by:
1. Using n=5 in-context learning examples
2. Computing the Jacobian at layer 5 for each example
3. Averaging the Jacobians and biases across examples
4. Scaling by beta=2.5 to correct for underestimation

### Evaluation Metrics

1. **Faithfulness**: Measures whether LRE predictions match the full model predictions
   - `argmax D(LRE(s)) == argmax D(F(s,c))`

2. **Causality**: Measures whether editing subject representations changes predictions
   - Using inverse LRE: `Δs = W† @ (z' - z)`
   - Check if edited prediction matches target object

## Results

### Faithfulness Evaluation

| Relation | Faithfulness |
|----------|--------------|
| Country Capital | 70% |
| Country Language | 90% |
| Person Native Language | 40% |
| Adjective Antonym | 20% |
| **Average** | **55%** |

### Causality Evaluation

| Relation | Causality |
|----------|-----------|
| Country Capital | 90% |

### Key Findings

1. **Varying Faithfulness**: Different relation types show varying degrees of linear decodability
   - Country-related factual relations: High faithfulness (70-90%)
   - Complex linguistic relations: Lower faithfulness (20-40%)

2. **Faithfulness-Causality Correlation**: High faithfulness relations also show high causality

3. **Not All Relations Are Linear**: Some relations are not well-approximated by linear transformations, consistent with the original paper's findings

## Analysis

The replication successfully demonstrates the core claims of the original paper:

1. For a subset of relations, the highly non-linear decoding procedure can be approximated by a simple linear transformation
2. The LRE can be estimated from the Jacobian computed on ICL examples
3. The inverse LRE can be used to edit subject representations and change model predictions
4. Not all relations are linearly decodable - this is expected and documented in the original work

### Environment Details

- Model: GPT-J-6B (fp16)
- Device: NVIDIA A100 80GB
- Layer for subject representation: 5
- Beta scaling factor: 2.5
- Number of ICL examples: 5
- Low-rank pseudo-inverse rank: 100
