#!/usr/bin/env python3
"""
Replication Script: Linearity of Relation Decoding in Transformer LMs

This script replicates the core experiment from the paper, demonstrating
that relation decoding can be approximated by linear transformations.
"""

import os
import sys
import json
import random
from datetime import datetime

# Set working directory
os.chdir('/net/scratch2/smallyan/relations_eval')
sys.path.insert(0, '/net/scratch2/smallyan/relations_eval')

import torch
import numpy as np

# Set seeds for reproducibility
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 70)
print("REPLICATION: Linearity of Relation Decoding in Transformer LMs")
print("=" * 70)

# Import repository modules
from src import models, data, functional, lens
from src.operators import JacobianIclMeanEstimator
from src.editors import LowRankPInvEditor
from src.utils import experiment_utils

# =============================================================================
# Part 1: Load Model
# =============================================================================
print("\n[1/7] Loading GPT-J model...")
mt = models.load_model("gptj", device=DEVICE, fp16=True)
print(f"Model dtype: {mt.model.dtype}")
print(f"Model device: {mt.model.device}")
print(f"Memory: {mt.model.get_memory_footprint() / 1e9:.2f} GB")

# =============================================================================
# Part 2: Load Dataset
# =============================================================================
print("\n[2/7] Loading dataset...")
dataset = data.load_dataset()
print(f"Loaded {len(dataset)} relations")

# Select the country capital city relation (as in demo)
relation_name = "country capital city"
relation = dataset.filter(relation_names=[relation_name])[0]
print(f"\nSelected relation: {relation.name}")
print(f"Number of samples: {len(relation.samples)}")
print(f"Prompt template: {relation.prompt_templates[0]}")

# =============================================================================
# Part 3: Split Data
# =============================================================================
print("\n[3/7] Splitting data...")
experiment_utils.set_seed(SEED)
train_relation, test_relation = relation.split(5)

print(f"Training samples ({len(train_relation.samples)}):")
for s in train_relation.samples:
    print(f"  {s}")

print(f"\nTest samples ({len(test_relation.samples)}):")
for s in test_relation.samples[:5]:
    print(f"  {s}")
if len(test_relation.samples) > 5:
    print(f"  ... and {len(test_relation.samples) - 5} more")

# =============================================================================
# Part 4: Estimate LRE
# =============================================================================
print("\n[4/7] Estimating LRE operator...")

# Hyperparameters matching the demo
H_LAYER = 5
BETA = 2.5
RANK = 100

print(f"Hyperparameters: h_layer={H_LAYER}, beta={BETA}, rank={RANK}")

estimator = JacobianIclMeanEstimator(
    mt=mt,
    h_layer=H_LAYER,
    beta=BETA
)

lre_operator = estimator(relation.set(samples=train_relation.samples))
print(f"LRE estimated using prompt template: {lre_operator.prompt_template[:50]}...")
print(f"Weight matrix shape: {lre_operator.weight.shape}")
print(f"Bias shape: {lre_operator.bias.shape}")

# =============================================================================
# Part 5: Filter Test Samples
# =============================================================================
print("\n[5/7] Filtering test samples...")
filtered_test = functional.filter_relation_samples_based_on_provided_fewshots(
    mt=mt,
    test_relation=test_relation,
    prompt_template=lre_operator.prompt_template,
    batch_size=4
)
print(f"Filtered test samples: {len(filtered_test.samples)} (from {len(test_relation.samples)})")

# =============================================================================
# Part 6: Evaluate Faithfulness
# =============================================================================
print("\n[6/7] Evaluating faithfulness...")
print("-" * 70)

correct = 0
wrong = 0

for sample in filtered_test.samples:
    predictions = lre_operator(subject=sample.subject).predictions
    top_pred = predictions[0]

    is_correct = functional.is_nontrivial_prefix(
        prediction=top_pred.token,
        target=sample.object
    )

    marker = "Y" if is_correct else "X"
    pred_str = functional.format_whitespace(top_pred.token)
    print(f"[{marker}] {sample.subject} -> {sample.object}, "
          f"predicted: '{pred_str}' (p={top_pred.prob:.3f})")

    if is_correct:
        correct += 1
    else:
        wrong += 1

faithfulness = correct / (correct + wrong) if (correct + wrong) > 0 else 0
print("-" * 70)
print(f"Faithfulness: {faithfulness:.2%} ({correct}/{correct + wrong})")

# =============================================================================
# Part 7: Evaluate Causality
# =============================================================================
print("\n[7/7] Evaluating causality...")

# Create editor
svd = torch.svd(lre_operator.weight.float())
editor = LowRankPInvEditor(
    lre=lre_operator,
    rank=RANK,
    svd=svd
)

# Generate random edit targets
experiment_utils.set_seed(SEED)
test_targets = functional.random_edit_targets(filtered_test.samples)
print(f"Generated {len(test_targets)} edit targets")

print("-" * 70)

success = 0
fails = 0

for sample in filtered_test.samples:
    target = test_targets.get(sample)
    if target is None:
        continue

    edit_result = editor(
        subject=sample.subject,
        target=target.subject
    )

    top_pred = edit_result.predicted_tokens[0]

    is_success = functional.is_nontrivial_prefix(
        prediction=top_pred.token,
        target=target.object
    )

    marker = "Y" if is_success else "X"
    print(f"[{marker}] {sample.subject} -> {target.object}, "
          f"predicted: '{top_pred.token}' (p={top_pred.prob:.3f})")

    if is_success:
        success += 1
    else:
        fails += 1

causality = success / (success + fails) if (success + fails) > 0 else 0
print("-" * 70)
print(f"Causality: {causality:.2%} ({success}/{success + fails})")

# =============================================================================
# Results Summary
# =============================================================================
print("\n" + "=" * 70)
print("REPLICATION RESULTS SUMMARY")
print("=" * 70)

print(f"\nRelation: {relation.name}")
print(f"Model: GPT-J-6B")

print(f"\nHyperparameters:")
print(f"  h_layer: {H_LAYER}")
print(f"  beta: {BETA}")
print(f"  rank (for causality): {RANK}")

print(f"\nData Split:")
print(f"  Training samples: {len(train_relation.samples)}")
print(f"  Test samples (filtered): {len(filtered_test.samples)}")

print(f"\nResults:")
print(f"  Faithfulness: {faithfulness:.2%}")
print(f"  Causality: {causality:.2%}")

# Reference values from original demo
print(f"\nOriginal Demo Reference:")
print(f"  Faithfulness: ~78.9%")
print(f"  Causality: ~100%")

# Check if results match expectations
faith_diff = abs(faithfulness - 0.789)
caus_diff = abs(causality - 1.0)

print(f"\nComparison:")
print(f"  Faithfulness difference from demo: {faith_diff:.1%}")
print(f"  Causality difference from demo: {caus_diff:.1%}")

faith_match = faith_diff < 0.20  # Allow some variance due to different random splits
caus_match = causality >= 0.80

print(f"\nReplication Status:")
print(f"  Faithfulness in expected range (within 20%): {'PASS' if faith_match else 'FAIL'}")
print(f"  Causality in expected range (>=80%): {'PASS' if caus_match else 'FAIL'}")
print(f"  Overall: {'SUCCESS' if faith_match and caus_match else 'PARTIAL/NEEDS REVIEW'}")

# Save results
results = {
    "timestamp": datetime.now().isoformat(),
    "relation": relation.name,
    "model": "GPT-J-6B",
    "hyperparameters": {
        "h_layer": H_LAYER,
        "beta": BETA,
        "rank": RANK
    },
    "data": {
        "train_samples": len(train_relation.samples),
        "test_samples": len(filtered_test.samples)
    },
    "results": {
        "faithfulness": faithfulness,
        "causality": causality
    },
    "reference": {
        "faithfulness": 0.789,
        "causality": 1.0
    },
    "replication_status": {
        "faithfulness_match": faith_match,
        "causality_match": caus_match,
        "overall_success": faith_match and caus_match
    }
}

results_file = '/net/scratch2/smallyan/relations_eval/evaluation/replications/replication_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {results_file}")

print("\n" + "=" * 70)
print("REPLICATION COMPLETE")
print("=" * 70)
