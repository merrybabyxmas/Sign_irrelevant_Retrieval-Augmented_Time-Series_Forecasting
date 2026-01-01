#!/usr/bin/env python3
"""
Visual comparison of different similarity metrics for RAFT time-series retrieval.

This script demonstrates the differences between:
1. Standard cosine similarity
2. Phase-aware cosine similarity (k=4)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def cosine_similarity(x, y):
    """Standard cosine similarity"""
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return (x_norm * y_norm).sum(dim=-1)

def phase_aware_similarity(x, y, k=4):
    """Phase-aware cosine similarity: ρ = cos(k * θ)"""
    eps = 1e-8
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    cos_sim = (x_norm * y_norm).sum(dim=-1)
    cos_sim_clamped = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_sim_clamped)
    return torch.cos(k * theta)

def plot_similarity_comparison():
    """Plot similarity metrics across different angles"""

    # Create angles from 0 to π
    angles = np.linspace(0, np.pi, 1000)

    # Create reference vector
    x = torch.tensor([[1.0, 0.0]])

    # Compute similarities at each angle
    cos_sims = []
    phase_sims_k2 = []
    phase_sims_k4 = []
    phase_sims_k8 = []

    for angle in angles:
        y = torch.tensor([[np.cos(angle), np.sin(angle)]])
        cos_sims.append(cosine_similarity(x, y).item())
        phase_sims_k2.append(phase_aware_similarity(x, y, k=2).item())
        phase_sims_k4.append(phase_aware_similarity(x, y, k=4).item())
        phase_sims_k8.append(phase_aware_similarity(x, y, k=8).item())

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Similarity Metrics for Time-Series Retrieval', fontsize=16, fontweight='bold')

    # Plot 1: Standard vs Phase-aware (k=4)
    ax = axes[0, 0]
    ax.plot(np.degrees(angles), cos_sims, 'b-', linewidth=2, label='Standard cosine')
    ax.plot(np.degrees(angles), phase_sims_k4, 'r-', linewidth=2, label='Phase-aware (k=4)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Angle θ (degrees)', fontsize=11)
    ax.set_ylabel('Similarity', fontsize=11)
    ax.set_title('Standard vs Phase-Aware (k=4)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 180)
    ax.set_ylim(-1.1, 1.1)

    # Add annotations
    ax.annotate('Both high at θ=0°\n(aligned)', xy=(0, 1), xytext=(20, 0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=9, color='green', fontweight='bold')
    ax.annotate('Phase-aware high at θ=180°\n(opposite but same shape)',
                xy=(180, 1), xytext=(140, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', fontweight='bold')

    # Plot 2: Different k values
    ax = axes[0, 1]
    ax.plot(np.degrees(angles), cos_sims, 'b-', linewidth=2, label='k=1 (standard)')
    ax.plot(np.degrees(angles), phase_sims_k2, 'g-', linewidth=2, label='k=2')
    ax.plot(np.degrees(angles), phase_sims_k4, 'r-', linewidth=2, label='k=4 (proposed)')
    ax.plot(np.degrees(angles), phase_sims_k8, 'm-', linewidth=2, label='k=8')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Angle θ (degrees)', fontsize=11)
    ax.set_ylabel('Similarity', fontsize=11)
    ax.set_title('Effect of Frequency Multiplier k', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 180)
    ax.set_ylim(-1.1, 1.1)

    # Plot 3: Time-series examples - aligned
    ax = axes[1, 0]
    t = np.linspace(0, 4*np.pi, 100)
    signal1 = np.sin(t)
    signal2_aligned = np.sin(t)
    signal2_opposite = -np.sin(t)

    ax.plot(t, signal1, 'b-', linewidth=2, label='Query signal', alpha=0.7)
    ax.plot(t, signal2_aligned, 'g--', linewidth=2, label='Aligned (θ=0°)', alpha=0.7)
    ax.plot(t, signal2_opposite, 'r:', linewidth=2, label='Opposite (θ=180°)', alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title('Time-Series Pattern Examples', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Add text box with similarity scores
    textstr = 'Standard Cosine:\n  Aligned: +1.0\n  Opposite: -1.0\n\nPhase-Aware (k=4):\n  Aligned: +1.0\n  Opposite: +1.0'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.5, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='right', bbox=props)

    # Plot 4: Comparison table
    ax = axes[1, 1]
    ax.axis('off')

    # Create comparison table
    table_data = [
        ['Angle θ', 'Standard\ncos(θ)', 'Phase-Aware\ncos(4θ)', 'Interpretation'],
        ['0° (aligned)', '+1.00', '+1.00', 'Both detect match'],
        ['45°', '+0.71', '-1.00', 'Phase-aware rejects'],
        ['90° (orthogonal)', '0.00', '+1.00', 'Phase-aware accepts'],
        ['135°', '-0.71', '-1.00', 'Phase-aware rejects'],
        ['180° (opposite)', '-1.00', '+1.00', 'Phase-aware accepts!'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.20, 0.20, 0.20, 0.40])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, 6):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    # Highlight the key insight
    table[(5, 3)].set_facecolor('#ffeb3b')
    table[(5, 3)].set_text_props(weight='bold')

    ax.text(0.5, 0.05, 'Key Insight: Phase-aware similarity treats opposite patterns as similar\n(same shape, different sign)',
            transform=ax.transAxes, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/home/dongwoo38/RAFT/similarity_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: /home/dongwoo38/RAFT/similarity_comparison.png")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Generating similarity comparison visualization...")
    print("="*70 + "\n")

    plot_similarity_comparison()

    print("\n" + "="*70)
    print("Key Takeaways:")
    print("="*70)
    print("1. Standard cosine: treats opposite patterns as dissimilar (-1.0)")
    print("2. Phase-aware (k=4): treats opposite patterns as similar (+1.0)")
    print("3. Phase-aware: rejects intermediate misalignments (45°, 135°)")
    print("4. Phase-aware: accepts orthogonal patterns (90°)")
    print()
    print("For time-series retrieval, shape matters more than sign!")
    print("="*70 + "\n")
