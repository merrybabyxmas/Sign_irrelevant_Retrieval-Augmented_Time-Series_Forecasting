# Phase-Aware Similarity Implementation for RAFT

## Summary

This document describes the implementation of **phase-aware cosine similarity** for time-series retrieval in the RAFT model. The modification allows for clean experimental comparison between:

1. **Pearson correlation** (original)
2. **Standard cosine similarity** (original)
3. **Phase-aware cosine similarity** (new) - `cos(k·θ)`

---

## Mathematical Background

### Original RAFT Similarity

In the original implementation, similarity between query patch `x` and key patches `k_i` is computed using Pearson correlation or cosine similarity:

```
ρ_i = cos(θ_i)

where cos(θ_i) = (x^T · k_i) / (||x|| · ||k_i||)
```

This treats:
- **Perfectly aligned vectors** (θ=0) as maximally similar
- **Perfectly anti-aligned vectors** (θ=π) as maximally dissimilar

However, in time-series retrieval, **shape similarity matters more than sign**, and many patterns are periodic or phase-shifted.

### Phase-Aware Similarity (New)

The new similarity metric is defined as:

```
ρ_i = cos(k · arccos(cos(θ_i))) = cos(k · θ_i)
```

where:
- `cos(θ_i)` is the standard cosine similarity
- `k` is an integer frequency multiplier (default: k=4)
- `θ_i` is the angle between vectors

**Key Properties:**
- Periodic with period `π/2` (for k=4)
- Multiple maxima at: `θ ∈ {0, π/2, π, 3π/2, 2π}`
- Intermediate phase misalignment is strongly penalized
- Output range: `[-1, 1]`
- Fully differentiable
- **When k=1**: Recovers standard cosine similarity exactly

---

## Implementation Changes

### Modified Files

1. **`layers/Retrieval.py`**
   - Added `similarity_type` parameter: `'cosine'`, `'pearson'`, or `'phase_aware'`
   - Added `phase_multiplier` parameter: integer k value (default: 4)
   - Modified `periodic_batch_corr()` method to compute phase-aware similarity

2. **`models/RAFT.py`**
   - Updated `RetrievalTool` instantiation to pass new parameters from config

3. **`run.py`**
   - Added `--similarity_type` command-line argument
   - Added `--phase_multiplier` command-line argument

---

## Usage

### Running with Default (Original) Cosine Similarity

```bash
python run.py \
  --data ETTh1 \
  --root_path ./data \
  --data_path ETTh1.csv
```

### Running with Phase-Aware Similarity (k=4)

```bash
python run.py \
  --data ETTh1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --similarity_type phase_aware \
  --phase_multiplier 4
```

### Running with Pearson Correlation

```bash
python run.py \
  --data ETTh1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --similarity_type pearson
```

### Ablation Study: Different k Values

```bash
# Standard cosine (equivalent to k=1)
python run.py --data ETTh1 --similarity_type phase_aware --phase_multiplier 1

# Phase-aware with k=2
python run.py --data ETTh1 --similarity_type phase_aware --phase_multiplier 2

# Phase-aware with k=4 (recommended default)
python run.py --data ETTh1 --similarity_type phase_aware --phase_multiplier 4

# Phase-aware with k=8
python run.py --data ETTh1 --similarity_type phase_aware --phase_multiplier 8
```

---

## Code Location

The core similarity computation is in `layers/Retrieval.py`, lines 112-136:

```python
# Compute similarity based on selected method
if self.similarity_type == 'pearson' or self.similarity_type == 'cosine':
    # Original implementation: Pearson correlation / cosine similarity
    # Both are equivalent after centering (mean removal)
    cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))

elif self.similarity_type == 'phase_aware':
    # Phase-aware cosine similarity for time-series retrieval
    # This treats shape similarity independently of sign/phase
    eps = 1e-8

    # Compute standard cosine similarity: cos(θ)
    bx_norm = F.normalize(bx, dim=2)
    ax_norm = F.normalize(ax, dim=2)
    cos_sim = torch.bmm(bx_norm, ax_norm.transpose(-1, -2))

    # Compute angle θ from cosine similarity
    # Clamp to avoid numerical issues with acos
    cos_sim_clamped = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_sim_clamped)

    # Apply phase-aware transformation: ρ = cos(k * θ)
    # This creates multiple maxima at θ ∈ {0, π/2, π, 3π/2}
    # and penalizes intermediate phase misalignments
    cur_sim = torch.cos(self.phase_multiplier * theta)
```

---

## Verification

Run the verification script to confirm correctness:

```bash
cd /home/dongwoo38/RAFT
python verify_similarity.py
```

**All tests should pass**, confirming:
1. ✓ k=1 recovers standard cosine similarity (max diff < 1e-5)
2. ✓ Output range is [-1, 1]
3. ✓ Expected behavior at key angles (0°, 90°, 180°)
4. ✓ Numerical stability (no NaN/Inf)

---

## Switching Between Similarity Methods

### Option 1: Command-Line Arguments (Recommended)

Simply pass `--similarity_type` and `--phase_multiplier` when running:

```bash
# Original cosine
python run.py --data ETTh1 --similarity_type cosine

# Phase-aware
python run.py --data ETTh1 --similarity_type phase_aware --phase_multiplier 4
```

### Option 2: Modify Default in `run.py`

Change line 97 in `run.py`:

```python
# For phase-aware by default
parser.add_argument('--similarity_type', type=str, default='phase_aware', ...)

# For original cosine by default
parser.add_argument('--similarity_type', type=str, default='cosine', ...)
```

---

## Design Principles Followed

✅ **Minimal changes**: Only modified similarity computation
✅ **No refactoring**: Kept all existing code structure
✅ **No new modules**: Implemented within existing `Retrieval.py`
✅ **Backward compatible**: Default behavior unchanged
✅ **No architectural changes**: Retrieval logic, top-m selection, softmax weighting, temperature scaling all unchanged
✅ **No training changes**: Loss, optimizer, training loop untouched
✅ **No evaluation changes**: Metrics remain the same
✅ **No new dependencies**: Uses only existing PyTorch functions
✅ **Well documented**: Clear comments explaining the phase-aware similarity logic

---

## Experimental Comparison

The implementation enables clean ablation studies for research papers:

| Similarity Type | Command | Description |
|----------------|---------|-------------|
| Pearson | `--similarity_type pearson` | Original (centered correlation) |
| Cosine | `--similarity_type cosine` | Original (normalized dot product) |
| Phase-aware (k=1) | `--similarity_type phase_aware --phase_multiplier 1` | Equivalent to cosine (sanity check) |
| Phase-aware (k=4) | `--similarity_type phase_aware --phase_multiplier 4` | Proposed method |

All experiments use identical:
- Model architecture
- Training procedure
- Evaluation metrics
- Hyperparameters (except similarity type)

This ensures differences in results are solely due to the similarity metric.

---

## Technical Notes

### Numerical Stability

- Uses `eps = 1e-8` for clamping to avoid `acos` domain errors
- Clamps cosine similarity to `[-1 + eps, 1 - eps]` before `acos`
- All operations are differentiable for gradient-based training

### Performance

- No significant computational overhead compared to original cosine similarity
- Both methods require:
  1. Normalization
  2. Batch matrix multiplication
- Phase-aware adds:
  1. One `acos` operation
  2. One `cos` operation
  3. One scalar multiplication

These are vectorized PyTorch operations with negligible overhead.

### Shape Preservation

- Tensor shapes are identical to original implementation
- Batch processing logic unchanged
- Memory footprint identical

---

## Contact & Attribution

Implementation by: Claude Sonnet 4.5
Date: 2025-12-30
Based on requirements specified by: dongwoo38

For questions about this implementation, refer to this document or the inline code comments in `layers/Retrieval.py`.
