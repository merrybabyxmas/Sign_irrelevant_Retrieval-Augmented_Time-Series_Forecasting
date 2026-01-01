#!/usr/bin/env python3
"""
Visualization script for sign-attenuated phase-aware similarity.

This script creates plots showing how different attenuation factors
affect similarity across different angles.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cosine_similarity(x, y):
    """Standard cosine similarity"""
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return (x_norm * y_norm).sum(dim=-1)

def phase_aware_similarity(x, y, k=4, neg_sign_weight=1.0):
    """Sign-attenuated phase-aware similarity"""
    eps = 1e-8
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    cos_sim = (x_norm * y_norm).sum(dim=-1)
    cos_sim_clamped = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_sim_clamped)
    phase_sim = torch.cos(k * theta)

    sign_weight = torch.where(
        cos_sim >= 0,
        torch.ones_like(cos_sim),
        neg_sign_weight * torch.ones_like(cos_sim)
    )

    return phase_sim * sign_weight

def plot_comprehensive_comparison():
    """Create comprehensive visualization"""

    # Create angles from 0 to π
    angles = np.linspace(0, np.pi, 1000)
    x = torch.tensor([[1.0, 0.0]])

    # Compute similarities
    cos_sims = []
    phase_sims = {1.0: [], 0.8: [], 0.6: [], 0.4: []}

    for angle in angles:
        y = torch.tensor([[np.cos(angle), np.sin(angle)]])
        cos_sims.append(cosine_similarity(x, y).item())

        for lambda_val in [1.0, 0.8, 0.6, 0.4]:
            sim = phase_aware_similarity(x, y, k=4, neg_sign_weight=lambda_val).item()
            phase_sims[lambda_val].append(sim)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('Sign-Attenuated Phase-Aware Similarity (k=4)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: All attenuation factors
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(np.degrees(angles), cos_sims, 'k--', linewidth=2,
             label='Standard cosine', alpha=0.5)

    colors = {1.0: 'blue', 0.8: 'green', 0.6: 'orange', 0.4: 'red'}
    for lambda_val in [1.0, 0.8, 0.6, 0.4]:
        label = f'λ_neg={lambda_val:.1f}' + (' (original)' if lambda_val == 1.0 else '')
        ax1.plot(np.degrees(angles), phase_sims[lambda_val],
                linewidth=2.5, label=label, color=colors[lambda_val])

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=90, color='gray', linestyle=':', alpha=0.3, label='θ=90° (cos=0)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Angle θ (degrees)', fontsize=12)
    ax1.set_ylabel('Similarity', fontsize=12)
    ax1.set_title('Effect of Sign Attenuation Factor λ_neg', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_xlim(0, 180)
    ax1.set_ylim(-1.1, 1.1)

    # Highlight attenuation region
    ax1.axvspan(90, 180, alpha=0.1, color='red', label='Attenuation region (cos<0)')

    # Plot 2: Cosine similarity reference
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(np.degrees(angles), cos_sims, 'b-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=90, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Angle θ (degrees)', fontsize=11)
    ax2.set_ylabel('cos(θ)', fontsize=11)
    ax2.set_title('Standard Cosine Similarity', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 180)
    ax2.set_ylim(-1.1, 1.1)
    ax2.fill_between(np.degrees(angles), cos_sims, 0,
                     where=np.array(cos_sims) < 0, alpha=0.2, color='red',
                     label='Negative region')
    ax2.legend(fontsize=9)

    # Plot 3: Phase similarity (no attenuation)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(np.degrees(angles), phase_sims[1.0], 'b-', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=90, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Angle θ (degrees)', fontsize=11)
    ax3.set_ylabel('cos(4θ)', fontsize=11)
    ax3.set_title('Phase-Aware (λ_neg=1.0, original)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 180)
    ax3.set_ylim(-1.1, 1.1)

    # Plot 4: Attenuation effect at key angles
    ax4 = fig.add_subplot(gs[2, 0])
    key_angles = [0, 45, 90, 135, 180]
    lambda_values = [1.0, 0.8, 0.6, 0.4]

    x_pos = np.arange(len(key_angles))
    width = 0.2

    for i, lambda_val in enumerate(lambda_values):
        values = []
        for angle_deg in key_angles:
            angle_rad = np.radians(angle_deg)
            y = torch.tensor([[np.cos(angle_rad), np.sin(angle_rad)]])
            sim = phase_aware_similarity(x, y, k=4, neg_sign_weight=lambda_val).item()
            values.append(sim)

        offset = (i - 1.5) * width
        ax4.bar(x_pos + offset, values, width,
               label=f'λ={lambda_val:.1f}', color=colors[lambda_val])

    ax4.set_xlabel('Angle θ (degrees)', fontsize=11)
    ax4.set_ylabel('Similarity', fontsize=11)
    ax4.set_title('Similarity at Key Angles', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{a}°' for a in key_angles])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Plot 5: Comparison table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    table_data = [
        ['Angle', 'cos(θ)', 'λ=1.0', 'λ=0.8', 'λ=0.6', 'λ=0.4'],
        ['0° (aligned)', '+1.00', '+1.00', '+1.00', '+1.00', '+1.00'],
        ['45°', '+0.71', '-1.00', '-1.00', '-1.00', '-1.00'],
        ['90° (orthog)', '0.00', '+1.00', '+1.00', '+1.00', '+1.00'],
        ['135°', '-0.71', '-1.00', '-0.80', '-0.60', '-0.40'],
        ['180° (opp)', '-1.00', '+1.00', '+0.80', '+0.60', '+0.40'],
    ]

    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.20, 0.16, 0.16, 0.16, 0.16, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, 6):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    # Highlight attenuation effects
    for i in [4, 5]:  # 135° and 180° rows
        for j in [3, 4, 5]:  # λ=0.8, 0.6, 0.4 columns
            table[(i, j)].set_facecolor('#ffeb3b')

    ax5.text(0.5, 0.08,
            'Yellow cells: Sign attenuation reduces similarity for anti-correlated patterns (cos<0)',
            transform=ax5.transAxes, fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.savefig('/home/dongwoo38/RAFT/sign_attenuation_visualization.png',
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: /home/dongwoo38/RAFT/sign_attenuation_visualization.png")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Generating sign-attenuation visualization...")
    print("="*70 + "\n")

    plot_comprehensive_comparison()

    print("\n" + "="*70)
    print("Key Insights:")
    print("="*70)
    print("1. When cos(θ) ≥ 0: All λ_neg values give identical results")
    print("2. When cos(θ) < 0: Similarity is scaled by λ_neg")
    print("3. At θ=180° (opposite patterns):")
    print("   - λ=1.0: similarity = +1.0 (same as aligned)")
    print("   - λ=0.6: similarity = +0.6 (40% attenuation)")
    print("   - λ=0.4: similarity = +0.4 (60% attenuation)")
    print()
    print("Use lower λ_neg when sign/direction matters in your application!")
    print("="*70 + "\n")
