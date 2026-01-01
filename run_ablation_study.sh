#!/bin/bash

# Ablation study script for sign-attenuated phase-aware similarity
#
# This script runs experiments with different neg_sign_weight values
# to understand the impact of sign attenuation on forecasting performance

DATA="ETTm1"
ROOT_PATH="./data"
DATA_PATH="ETTm1.csv"

echo "========================================================================"
echo "RAFT Sign-Attenuation Ablation Study"
echo "========================================================================"
echo ""
echo "Dataset: ${DATA}"
echo "This will run 5 experiments with different attenuation factors"
echo ""

# Experiment 1: Baseline (standard cosine similarity)
echo "========================================================================"
echo "Experiment 1/5: Baseline (Standard Cosine Similarity)"
echo "========================================================================"
python run.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --similarity_type cosine \
  --des "baseline_cosine"

echo ""
echo "✓ Experiment 1 completed"
echo ""

# Experiment 2: Phase-aware (original, λ=1.0)
echo "========================================================================"
echo "Experiment 2/5: Phase-Aware (Original, λ_neg=1.0)"
echo "========================================================================"
python run.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --similarity_type phase_aware \
  --phase_multiplier 4 \
  --neg_sign_weight 1.0 \
  --des "phase_aware_lambda1.0"

echo ""
echo "✓ Experiment 2 completed"
echo ""

# Experiment 3: Sign-attenuated (λ=0.8)
echo "========================================================================"
echo "Experiment 3/5: Sign-Attenuated (λ_neg=0.8)"
echo "========================================================================"
python run.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --similarity_type phase_aware \
  --phase_multiplier 4 \
  --neg_sign_weight 0.8 \
  --des "phase_aware_lambda0.8"

echo ""
echo "✓ Experiment 3 completed"
echo ""

# Experiment 4: Sign-attenuated (λ=0.6)
echo "========================================================================"
echo "Experiment 4/5: Sign-Attenuated (λ_neg=0.6)"
echo "========================================================================"
python run.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --similarity_type phase_aware \
  --phase_multiplier 4 \
  --neg_sign_weight 0.6 \
  --des "phase_aware_lambda0.6"

echo ""
echo "✓ Experiment 4 completed"
echo ""

# Experiment 5: Sign-attenuated (λ=0.4)
echo "========================================================================"
echo "Experiment 5/5: Sign-Attenuated (λ_neg=0.4)"
echo "========================================================================"
python run.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --similarity_type phase_aware \
  --phase_multiplier 4 \
  --neg_sign_weight 0.4 \
  --des "phase_aware_lambda0.4"

echo ""
echo "✓ Experiment 5 completed"
echo ""

echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================================================"
echo ""
echo "Results saved in ./results/ and ./checkpoints/"
echo ""
echo "Experiments conducted:"
echo "  1. Baseline (cosine)"
echo "  2. Phase-aware (λ=1.0) - original"
echo "  3. Phase-aware (λ=0.8) - 20% attenuation"
echo "  4. Phase-aware (λ=0.6) - 40% attenuation"
echo "  5. Phase-aware (λ=0.4) - 60% attenuation"
echo ""
echo "To analyze results, compare test metrics across experiments."
echo "========================================================================"
