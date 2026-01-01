#!/bin/bash

# Example script to run RAFT with phase-aware similarity
#
# This demonstrates how to use the new phase-aware similarity implementation
# while keeping all other settings identical to your current training

echo "========================================================================"
echo "Running RAFT with Phase-Aware Similarity (k=4)"
echo "========================================================================"
echo ""
echo "This will use the same dataset and configuration as your current run,"
echo "but with phase-aware similarity: ρ = cos(4θ)"
echo ""
echo "Mixture alpha options:"
echo "  --mixture_alpha 0.0  : Pure phase-aware (default)"
echo "  --mixture_alpha 0.3  : 30% cosine + 70% phase-aware"
echo "  --mixture_alpha 0.5  : 50% cosine + 50% phase-aware"
echo "  --mixture_alpha 1.0  : Pure cosine"
echo ""

python run.py \
  --data ETTh1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --similarity_type phase_aware \
  --neg_sign_weight 1.0 \
  --shift_range 1 \
  --phase_multipliers 2 \
  --mixture_alpha 0.7

echo ""
echo "========================================================================"
echo "Training completed with phase-aware similarity!"
echo "========================================================================"
