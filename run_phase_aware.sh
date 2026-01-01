#!/bin/bash

# Example script to run RAFT with phase-aware similarity using MULTIPLE multipliers
#
# This demonstrates the Multi-Multiplier Retrieval Expansion.
# We use k=1 (low frequency, stable) and k=4 (high frequency, precise).

echo "========================================================================"
echo "Running RAFT with Multi-Multiplier Phase-Aware Similarity (k=1, k=4)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  --phase_multipliers 1 4  : Retrieves candidates using both k=1 and k=4"
echo "  --neg_sign_weight 1.0    : No attenuation for negative correlations"
echo "  --shift_range 1          : Shift invariance enabled (+/- 1 step)"
echo ""

python run.py \
  --data ETTh1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --similarity_type phase_aware \
  --neg_sign_weight 1.0 \
  --shift_range 1 \
  --phase_multipliers 1 4 \
  --mixture_alpha 0.5

echo ""
echo "========================================================================"
echo "Training completed with multi-multiplier phase-aware similarity!"
echo "========================================================================"
