#!/usr/bin/env python
"""
Replication of Linear Relational Embedding (LRE) Experiment

This script replicates the key experiments from:
"Linearity of Relation Decoding in Transformer LMs" (https://arxiv.org/abs/2308.09124)

Experiments replicated:
1. LRE Faithfulness Evaluation
2. LRE Causality Evaluation
"""

import os
import sys
import json
import random
import numpy as np
import torch
from datetime import datetime

# Add repo to path
repo_path = '/net/scratch2/smallyan/relations_eval'
sys.path.insert(0, repo_path)

from src import models, data, functional
from src.operators import JacobianIclMeanEstimator
from src.editors import LowRankPInvEditor
from src import lens

# Set seeds for reproducibility
def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed()

print("=" * 60)
print("LRE REPLICATION EXPERIMENT")
print("=" * 60)

# Configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Hyperparameters from the plan
LAYER = 15  # Layer for extracting subject representation (will try multiple)
BETA = 2.5  # Scaling factor
RANK = 100  # Rank for low-rank pseudo-inverse in causality
N_TRAIN = 5  # Number of training examples for LRE

# Results storage
results = {
    "timestamp": datetime.now().isoformat(),
    "model": "gpt2-xl",
    "hyperparameters": {
        "layer": LAYER,
        "beta": BETA,
        "rank": RANK,
        "n_train": N_TRAIN
    },
    "faithfulness_results": {},
    "causality_results": {},
    "summary": {}
}

print("\nLoading GPT-2-XL model...")
mt = models.load_model("gpt2-xl", device=device, fp16=True)
print(f"Model loaded: {type(mt.model).__name__}")
print(f"  - Layers: {mt.model.config.n_layer}")
print(f"  - Hidden size: {mt.model.config.n_embd}")

print("\nLoading dataset...")
dataset = data.load_dataset()
relation_names = [r.name for r in dataset.relations]
print(f"Dataset loaded with {len(relation_names)} relations")

# Select a subset of relations to test (mix of different types)
# Based on the plan, we have factual, commonsense, linguistic, and bias categories
# Using correct naming convention with spaces
test_relations = [
    "country capital city",  # factual
    "person plays instrument",  # factual
    "fruit inside color",  # commonsense
    "verb past tense",  # linguistic
    "name gender",  # bias
]

# Filter to relations that exist in the dataset
test_relations = [r for r in test_relations if r in relation_names]
print(f"Testing on {len(test_relations)} relations: {test_relations}")

print("\n" + "=" * 60)
print("EXPERIMENT 1: LRE FAITHFULNESS EVALUATION")
print("=" * 60)

all_faithfulness_scores = []

for relation_name in test_relations:
    print(f"\nProcessing relation: {relation_name}")
    set_seed()  # Reset seed for each relation

    relation = dataset.filter(relation_names=[relation_name])[0]
    print(f"  Samples: {len(relation.samples)}")

    if len(relation.samples) < N_TRAIN + 5:
        print(f"  Skipping - not enough samples")
        continue

    # Split into train and test
    train, test = relation.split(N_TRAIN)
    print(f"  Train: {len(train.samples)}, Test: {len(test.samples)}")

    # Create LRE estimator
    estimator = JacobianIclMeanEstimator(
        mt=mt,
        h_layer=LAYER,
        beta=BETA
    )

    # Estimate the LRE operator
    print("  Estimating LRE operator...")
    operator = estimator(relation.set(samples=train.samples))

    # Filter test samples based on model knowledge
    test_filtered = functional.filter_relation_samples_based_on_provided_fewshots(
        mt=mt,
        test_relation=test,
        prompt_template=operator.prompt_template,
        batch_size=4
    )
    print(f"  Filtered test samples: {len(test_filtered.samples)}")

    if len(test_filtered.samples) == 0:
        print(f"  Skipping - no valid test samples")
        continue

    # Evaluate faithfulness
    correct = 0
    total = 0

    for sample in test_filtered.samples[:20]:  # Limit to 20 for speed
        predictions = operator(subject=sample.subject).predictions
        is_correct = functional.is_nontrivial_prefix(
            prediction=predictions[0].token, target=sample.object
        )
        correct += is_correct
        total += 1

    faithfulness = correct / total if total > 0 else 0
    print(f"  Faithfulness: {faithfulness:.3f} ({correct}/{total})")

    results["faithfulness_results"][relation_name] = {
        "faithfulness": faithfulness,
        "correct": correct,
        "total": total
    }
    all_faithfulness_scores.append(faithfulness)

avg_faithfulness = np.mean(all_faithfulness_scores) if all_faithfulness_scores else 0
print(f"\nAverage Faithfulness: {avg_faithfulness:.3f}")
results["summary"]["avg_faithfulness"] = avg_faithfulness

print("\n" + "=" * 60)
print("EXPERIMENT 2: LRE CAUSALITY EVALUATION")
print("=" * 60)

all_causality_scores = []

for relation_name in test_relations:
    print(f"\nProcessing relation: {relation_name}")
    set_seed()

    relation = dataset.filter(relation_names=[relation_name])[0]

    if len(relation.samples) < N_TRAIN + 5:
        print(f"  Skipping - not enough samples")
        continue

    train, test = relation.split(N_TRAIN)

    # Create LRE estimator
    estimator = JacobianIclMeanEstimator(
        mt=mt,
        h_layer=LAYER,
        beta=BETA
    )

    operator = estimator(relation.set(samples=train.samples))

    # Filter test samples
    test_filtered = functional.filter_relation_samples_based_on_provided_fewshots(
        mt=mt,
        test_relation=test,
        prompt_template=operator.prompt_template,
        batch_size=4
    )

    if len(test_filtered.samples) < 2:
        print(f"  Skipping - not enough test samples for causality")
        continue

    # Get random edit targets
    test_targets = functional.random_edit_targets(test_filtered.samples)

    # Create editor
    svd = torch.svd(operator.weight.float())
    editor = LowRankPInvEditor(
        lre=operator,
        rank=RANK,
        svd=svd
    )

    # Precompute hidden states
    all_subjects = list(set([s.subject for s in test_filtered.samples] +
                           [test_targets[s].subject for s in test_filtered.samples if s in test_targets]))

    hs_and_zs = functional.compute_hs_and_zs(
        mt=mt,
        prompt_template=operator.prompt_template,
        subjects=all_subjects,
        h_layer=operator.h_layer,
        z_layer=-1,
        batch_size=2
    )

    # Evaluate causality
    success = 0
    total = 0

    for sample in list(test_filtered.samples)[:10]:  # Limit for speed
        target = test_targets.get(sample)
        if target is None:
            continue

        edit_result = editor(
            subject=sample.subject,
            target=target.subject
        )

        is_success = functional.is_nontrivial_prefix(
            prediction=edit_result.predicted_tokens[0].token,
            target=target.object
        )
        success += is_success
        total += 1

    causality = success / total if total > 0 else 0
    print(f"  Causality: {causality:.3f} ({success}/{total})")

    results["causality_results"][relation_name] = {
        "causality": causality,
        "success": success,
        "total": total
    }
    all_causality_scores.append(causality)

avg_causality = np.mean(all_causality_scores) if all_causality_scores else 0
print(f"\nAverage Causality: {avg_causality:.3f}")
results["summary"]["avg_causality"] = avg_causality

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Average Faithfulness: {avg_faithfulness:.3f}")
print(f"Average Causality: {avg_causality:.3f}")

# Save results
output_path = '/net/scratch2/smallyan/relations_eval/evaluation/replications/replication_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_path}")

print("\nReplication complete!")
