# Plan
## Objective
Investigate how transformer language models represent and decode relational knowledge, specifically testing whether relation decoding can be well-approximated by linear transformations on subject representations.

## Hypothesis
1. For a variety of relations, transformer LMs decode relational knowledge directly from subject entity representations at intermediate layers.
2. For each relation, the decoding procedure is approximately affine (linear relational embedding), expressed as LRE(s) = Wrs + br mapping subject s to object o.
3. These affine transformations can be computed directly from the LM Jacobian on a prompt expressing the relation (∂o/∂s).
4. Not all relations are linearly decodable; some relations are reliably predicted but do not exhibit linear relational embeddings.

## Methodology
1. Extract Linear Relational Embeddings (LREs) by computing the mean Jacobian W and bias b from n=8 examples using first-order Taylor approximation: W = E[∂F/∂s] and b = E[F(s,c) - (∂F/∂s)s], scaled by β to correct underestimation.
2. Evaluate LRE faithfulness by measuring whether LRE(s) makes the same next-token predictions as the full transformer: argmax D(F(s,c))t = argmax D(LRE(s))t.
3. Evaluate LRE causality by using the inverse LRE to edit subject representations (Δs = W†(o' - o)) and checking whether the edit changes model predictions to target object o'.
4. Test on GPT-J, GPT-2-XL, and LLaMA-13B using a manually curated dataset of 47 relations across factual, commonsense, linguistic, and bias categories with over 10k subject-object pairs.

## Experiments
### LRE Faithfulness Evaluation
- What varied: Relations (47 total across factual, commonsense, linguistic, and bias categories)
- Metric: Faithfulness: frequency that argmax D(LRE(s)) matches argmax D(F(s,c)) on first token
- Main result: 48% of relations achieved >60% faithfulness on GPT-J; LRE outperformed baselines (Identity, Translation, Linear Regression) across all relation types; some relations like Company CEO showed <6% faithfulness indicating non-linear decoding.

### LRE Causality Evaluation
- What varied: Relations and edit interventions (LRE-based vs. baselines: oracle s' substitution, embedding o', output o')
- Metric: Causality: success rate of o' = argmax D(F(s, cr | s := s + Δs))
- Main result: LRE causality closely matched oracle baseline across layers; strong correlation (R=0.84) between faithfulness and causality when hyperparameters optimized for causality; LRE causality typically exceeded faithfulness scores.

### Layer-wise LRE Performance
- What varied: Layer at which subject representation s is extracted (embedding through layer 27 in GPT-J)
- Metric: Faithfulness and causality scores per layer
- Main result: LRE faithfulness increases through intermediate layers then plummets at later layers, suggesting a mode switch where representations transition from encoding subject attributes to predicting next tokens; effect disappears when object immediately follows subject.

### Baseline Comparison
- What varied: Linear approximation methods: LRE(s), LRE(es), Linear Regression, Translation, Identity
- Metric: Faithfulness across factual, linguistic, bias, and commonsense relations
- Main result: LRE applied to enriched representations s outperformed all baselines; LRE(es) on embeddings showed poor performance highlighting importance of intermediate enrichment; both projection W and bias b terms necessary.

### Attribute Lens Application
- What varied: Prompts (standard vs. repetition-distracted vs. instruction-distracted)
- Metric: Recall@k (k=1,2,3) of correct object in D(LRE(h)) distribution
- Main result: Attribute lens revealed latent knowledge even when LM outputs falsehoods; on distracted prompts where LM predicts wrong answer (2-3% R@1), attribute lens recovered correct fact 54-63% R@1.

### Cross-Model Analysis
- What varied: Language models (GPT-J, GPT-2-XL, LLaMA-13B)
- Metric: Faithfulness and causality per relation
- Main result: LRE performance strongly correlated across models (GPT-J vs GPT-2-XL: R=0.85; GPT-J vs LLaMA-13B: R=0.71); similar patterns of which relations are linearly decodable across different model architectures and sizes.