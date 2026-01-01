# Sign-Attenuated Phase-Aware Similarity for RAFT

## Overview

This document describes the **sign-attenuated phase-aware similarity** extension to the RAFT time-series retrieval model. This feature allows fine-grained control over how anti-correlated patterns are treated in retrieval, enabling clean ablation studies for research.

---

## Mathematical Definition

### Standard Phase-Aware Similarity (Original)

```
ρ = cos(k·θ)

where:
  θ = arccos(cos_sim)
  cos_sim = (x^T · k) / (||x|| · ||k||)
  k = phase multiplier (default: 4)
```

### Sign-Attenuated Phase-Aware Similarity (New)

```
ρ = cos(k·θ) · sign_weight

where:
  sign_weight = 1                if cos(θ) ≥ 0
                λ_neg            if cos(θ) < 0

  λ_neg ∈ (0, 1] = attenuation coefficient
```

---

## Key Properties

### When cos(θ) ≥ 0 (Positively Correlated)
- **No attenuation applied**
- Behavior identical to standard phase-aware similarity
- `sign_weight = 1.0`

### When cos(θ) < 0 (Anti-Correlated)
- **Attenuation applied**
- Similarity scaled by `λ_neg`
- `sign_weight = λ_neg`

### Special Cases

| Configuration | Behavior |
|--------------|----------|
| `k=1, λ_neg=1.0` | Standard cosine similarity |
| `k=4, λ_neg=1.0` | Standard phase-aware (original) |
| `k=4, λ_neg=0.6` | Phase-aware with 40% attenuation of anti-correlated patterns |
| `k=4, λ_neg=0.0` | Would reject all anti-correlated patterns (not allowed, min=0.0+ε) |

---

## Implementation Details

### Modified Files

1. **`layers/Retrieval.py`**
   - Added `neg_sign_weight` parameter to `__init__()` (line 24)
   - Added validation: `0.0 < neg_sign_weight <= 1.0` (lines 51-53)
   - Modified `periodic_batch_corr()` to apply sign attenuation (lines 144-152)

2. **`models/RAFT.py`**
   - Updated `RetrievalTool` instantiation (line 42)
   - Passes `neg_sign_weight` from config with default 1.0

3. **`run.py`**
   - Added `--neg_sign_weight` command-line argument (lines 104-107)
   - Default value: `1.0` (backward compatible)

### Core Implementation (lines 144-152 in Retrieval.py)

```python
# Apply phase-aware transformation: ρ = cos(k * θ)
phase_sim = torch.cos(self.phase_multiplier * theta)

# Apply sign-dependent attenuation
# When cos(θ) < 0 (anti-correlated), attenuate the similarity by neg_sign_weight
sign_weight = torch.where(
    cos_sim >= 0,
    torch.ones_like(cos_sim),
    self.neg_sign_weight * torch.ones_like(cos_sim)
)

cur_sim = phase_sim * sign_weight
```

---

## Usage Examples

### 1. Standard Cosine Similarity (Baseline)

```bash
python run.py \
  --data ETTm1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --similarity_type cosine
```

### 2. Phase-Aware Similarity (Original)

```bash
python run.py \
  --data ETTm1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --similarity_type phase_aware \
  --phase_multiplier 4
```

Equivalent to:
```bash
python run.py \
  --data ETTm1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --similarity_type phase_aware \
  --phase_multiplier 4 \
  --neg_sign_weight 1.0  # ← explicit default
```

### 3. Sign-Attenuated Phase-Aware Similarity (New)

```bash
python run.py \
  --data ETTm1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --similarity_type phase_aware \
  --phase_multiplier 4 \
  --neg_sign_weight 0.6  # ← 40% attenuation
```

---

## Ablation Study Design

For a clean ablation study, run experiments with:

| Experiment | `similarity_type` | `phase_multiplier` | `neg_sign_weight` | Description |
|-----------|------------------|-------------------|------------------|-------------|
| **Baseline** | `cosine` | N/A | N/A | Standard cosine similarity |
| **Phase (original)** | `phase_aware` | `4` | `1.0` (default) | Original phase-aware |
| **Sign-Atten (λ=0.8)** | `phase_aware` | `4` | `0.8` | 20% attenuation |
| **Sign-Atten (λ=0.6)** | `phase_aware` | `4` | `0.6` | 40% attenuation |
| **Sign-Atten (λ=0.4)** | `phase_aware` | `4` | `0.4` | 60% attenuation |

All other hyperparameters (epochs, learning rate, batch size, etc.) should remain identical.

---

## Behavioral Analysis

### Example: Opposite Patterns (θ=180°)

For **opposite but same-shape patterns** (e.g., `x = [1, 2, 3]` vs `y = [-1, -2, -3]`):

- `cos(θ) = -1.0` (anti-correlated)
- `θ = π`
- `cos(4π) = 1.0`

**Similarity values:**

| `λ_neg` | `ρ = cos(4π) · λ_neg` | Interpretation |
|---------|----------------------|----------------|
| 1.0 | 1.0 | Opposite patterns treated as highly similar |
| 0.8 | 0.8 | Slight penalty for anti-correlation |
| 0.6 | 0.6 | Moderate penalty |
| 0.4 | 0.4 | Strong penalty |

### Example: Aligned Patterns (θ=0°)

For **aligned patterns** (e.g., `x = [1, 2, 3]` vs `y = [1, 2, 3]`):

- `cos(θ) = 1.0` (positively correlated)
- `θ = 0`
- `cos(0) = 1.0`

**Similarity values (independent of λ_neg):**

| `λ_neg` | `ρ = cos(0) · 1.0` | Interpretation |
|---------|-------------------|----------------|
| any | 1.0 | Always maximum similarity (no attenuation) |

---

## Backward Compatibility

### ✅ Guaranteed Compatibility

1. **Old scripts without `--neg_sign_weight`**
   - Default value: `1.0`
   - Behavior: **Identical** to original phase-aware similarity
   - No changes to results

2. **Old scripts with `--similarity_type cosine`**
   - `neg_sign_weight` parameter ignored (only applies to `phase_aware`)
   - Behavior: **Unchanged**

3. **All existing checkpoints and configs**
   - Fully compatible
   - No retraining required

### Example: Running Existing Script

Your current script:
```bash
python run.py \
  --data ETTm1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --similarity_type phase_aware \
  --phase_multiplier 1
```

**Will produce identical results** after this update, because:
- `neg_sign_weight` defaults to `1.0`
- No attenuation applied

---

## Verification

Run the verification script to confirm correctness:

```bash
cd /home/dongwoo38/RAFT
python verify_sign_attenuation.py
```

**Expected output:**
```
ALL TESTS PASSED ✓

The sign-attenuated phase-aware similarity implementation is:
  • Backward compatible (neg_sign_weight=1.0 = original)
  • Mathematically correct
  • Numerically stable
  • Properly attenuates negative cosine similarities
```

---

## When to Use This Feature

### Use `λ_neg < 1.0` when:

1. **You want to penalize anti-correlated patterns**
   - Even if they have the same shape
   - Prefer patterns with same sign/direction

2. **Testing sensitivity to pattern polarity**
   - Ablation study to understand impact of sign
   - Research question: "Does sign matter?"

3. **Domain knowledge suggests sign is important**
   - E.g., in stock prices: rising vs falling trends
   - E.g., in temperature: heating vs cooling

### Keep `λ_neg = 1.0` (default) when:

1. **Shape is more important than sign**
   - Periodic patterns (e.g., seasonal data)
   - Symmetric phenomena

2. **Replicating original RAFT behavior**
   - Comparing with published results
   - Backward compatibility required

---

## Design Rationale

### Why This Design?

1. **Minimal code changes**
   - Only 3 files modified
   - ~20 lines of new code
   - No architectural changes

2. **Fully backward compatible**
   - Default `λ_neg = 1.0` preserves original behavior
   - Old scripts work unchanged

3. **CLI-controllable**
   - No need to modify Python code
   - Easy to sweep in .sh scripts

4. **Numerically stable**
   - Validation ensures `0 < λ_neg ≤ 1`
   - No division or numerical issues

5. **Clean ablation studies**
   - Single parameter controls attenuation
   - All other logic unchanged

---

## Research Applications

### Hypothesis Testing

**Hypothesis:** "In time-series forecasting, pattern shape is more important than pattern sign."

**Experimental Design:**

| λ_neg | Expected Behavior |
|-------|------------------|
| 1.0 | If hypothesis true: best performance (sign doesn't matter) |
| 0.6 | If hypothesis false: best performance (sign matters) |

Compare performance metrics (MSE, MAE) across different `λ_neg` values to test hypothesis.

### Sensitivity Analysis

Sweep `λ_neg ∈ {1.0, 0.8, 0.6, 0.4, 0.2}` and plot:
- x-axis: `λ_neg`
- y-axis: test MSE

If performance degrades as `λ_neg` decreases → sign matters
If performance stable → sign doesn't matter

---

## Technical Notes

### Computational Overhead

**Negligible overhead** compared to original phase-aware similarity:

- Added operations:
  1. `torch.where()` - conditional selection (vectorized)
  2. Scalar multiplication

- Both are O(1) per element, fully GPU-accelerated
- No loops, no additional memory allocation

### Numerical Stability

- Validation ensures `0 < λ_neg ≤ 1` at initialization
- No risk of division by zero
- No risk of overflow/underflow
- All operations on normalized values

### Memory Footprint

- **Identical** to original implementation
- `sign_weight` tensor is temporary (garbage collected)
- No persistent memory overhead

---

## Common Questions

### Q: Does this affect training?

**A:** No. This only affects the **retrieval similarity computation**. The training loop, loss function, and optimizer are unchanged.

### Q: Can I use this with other similarity types?

**A:** No. `neg_sign_weight` only applies when `similarity_type='phase_aware'`. It's ignored for `cosine` and `pearson`.

### Q: What happens if I set `neg_sign_weight=0.0`?

**A:** Validation will fail with an assertion error. Valid range is `(0, 1]` (exclusive of 0, inclusive of 1).

### Q: How do I switch back to original behavior?

**A:** Either:
1. Omit `--neg_sign_weight` (uses default 1.0)
2. Explicitly set `--neg_sign_weight 1.0`

Both are equivalent.

---

## Summary

The sign-attenuated phase-aware similarity provides:

✅ **Backward compatibility** - Default behavior unchanged
✅ **Flexibility** - Fine-grained control via `λ_neg`
✅ **Simplicity** - Single CLI parameter
✅ **Stability** - Validated and tested
✅ **Research-ready** - Enables clean ablation studies

**No changes required to existing scripts** - all defaults preserve original behavior.

---

## Contact

Implementation: Claude Sonnet 4.5
Date: 2025-12-31
Based on requirements by: dongwoo38

For questions, refer to this document or inline code comments in `layers/Retrieval.py` lines 144-152.
