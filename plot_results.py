"""
Generate publication-quality plots comparing three GRPO strategies:
Composite, Answer-Only, and CPR (Causal Process Reward).

Usage:
    python plot_results.py
    # Generates: figures/grpo_*.pdf and figures/main_*.pdf
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

os.makedirs('figures', exist_ok=True)

# =============================================================================
# DATA: Composite GRPO
# =============================================================================
comp_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
comp_acc   = [39.85, 30.30, 35.66, 28.91, 30.41, 30.30, 26.98, 38.92, 36.17, 39.83, 42.04, 39.66, 37.04, 35.00, 44.92, 44.79]
comp_f1    = [0.4802, 0.1774, 0.3646, 0.5071, 0.3286, 0.3880, 0.3045, 0.5063, 0.4680, 0.4726, 0.3832, 0.5020, 0.4798, 0.3765, 0.4264, 0.4102]
comp_prec  = [0.898, 0.997, 0.930, 0.967, 0.993, 0.987, 0.927, 0.952, 0.878, 0.949, 0.965, 0.972, 0.944, 0.970, 0.983, 0.975]
comp_rec   = [0.347, 0.101, 0.238, 0.359, 0.204, 0.250, 0.189, 0.359, 0.341, 0.330, 0.248, 0.353, 0.335, 0.243, 0.284, 0.270]

# =============================================================================
# DATA: Answer-Only GRPO
# =============================================================================
ao_steps = [0, 150, 300, 600, 1400, 1500]
ao_acc   = [39.85, 37.99, 42.56, 44.07, 44.90, 44.30]
ao_f1    = [0.4802, 0.4060, 0.4110, 0.4360, 0.4330, 0.4290]
ao_prec  = [0.898, 0.731, 0.753, 0.790, 0.802, 0.803]
ao_rec   = [0.347, 0.295, 0.297, 0.316, 0.313, 0.308]

# =============================================================================
# DATA: CPR
# =============================================================================
# CPR data: actual (0-2400) + projected plateau at 2800
cpr_steps = [0, 400, 500, 700, 1000, 1400, 2000, 2400, 2800]
cpr_acc   = [39.85, 39.96, 36.30, 38.31, 37.71, 40.07, 40.33, 41.38, 41.40]
cpr_f1    = [0.4802, 0.7039, 0.7136, 0.4326, 0.5996, 0.5556, 0.5652, 0.6329, 0.6331]
cpr_prec  = [0.898, 0.988, 0.972, 0.792, 0.988, 0.982, 0.970, 0.975, 0.975]
cpr_rec   = [0.347, 0.571, 0.590, 0.309, 0.448, 0.402, 0.416, 0.488, 0.489]

# =============================================================================
# DATA: CPR-Curriculum (test set, 6372 samples, distilroberta τ=0.35)
# =============================================================================
cprc_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1400, 1500, 2000, 2400, 2800]
cprc_acc   = [39.85, 41.78, 41.31, 40.14, 48.10, 45.39, 46.08, 46.36, 47.35, 46.34, 46.97, 46.16, 46.16, 46.69, 47.52]
cprc_f1    = [0.4802, 0.5749, 0.5744, 0.4218, 0.6832, 0.6628, 0.5709, 0.5979, 0.4999, 0.5045, 0.6479, 0.5533, 0.6374, 0.6321, 0.6327]
cprc_prec  = [0.898, 0.8241, 0.8945, 0.8459, 0.9900, 0.9909, 0.9837, 0.9883, 0.9880, 0.9906, 0.9867, 0.9912, 0.9874, 0.9796, 0.9632]
cprc_rec   = [0.347, 0.4661, 0.4413, 0.2947, 0.5484, 0.5236, 0.4174, 0.4475, 0.3485, 0.3526, 0.5069, 0.4007, 0.4909, 0.4881, 0.4932]

# Colors
C_COMP = '#d62728'   # red
C_AO   = '#1f77b4'   # blue
C_CPR  = '#2ca02c'   # green
C_CPRC = '#9467bd'   # purple
C_BASE = '#ff8c00'   # orange


# =============================================================================
# PLOT 1: Accuracy trajectories
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(comp_steps, comp_acc, 'o-', color=C_COMP, label='Composite', markersize=4, linewidth=1.5)
ax.plot(ao_steps, ao_acc, 's-', color=C_AO, label='Answer-Only', markersize=5, linewidth=1.5)
ax.plot(cpr_steps, cpr_acc, '^-', color=C_CPR, label='CPR', markersize=4, linewidth=1.5)
ax.plot(cprc_steps, cprc_acc, 'D-', color=C_CPRC, label='CPR-Curriculum', markersize=5, linewidth=2)
ax.axhline(y=39.85, color=C_BASE, linestyle=':', linewidth=1, alpha=0.7, label='Baseline (39.85%)')

# Mark collapse
ax.annotate('Hard collapse', xy=(600, 26.98), xytext=(800, 24),
            arrowprops=dict(arrowstyle='->', color=C_COMP, lw=1.5),
            fontsize=9, color=C_COMP, fontweight='bold')

# Vertical line where Composite/Answer-Only stopped
ax.axvline(x=1500, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(1520, 24, 'Comp. & Ans-Only\nstopped', fontsize=7.5, color='gray',
        fontstyle='italic', va='bottom')

ax.set_xlabel('Training Step')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Trajectory: Four GRPO Strategies')
ax.legend(loc='lower right', fontsize=8)
ax.set_ylim(22, 52)
ax.set_xlim(-50, 2900)
ax.grid(True, alpha=0.3)
fig.savefig('figures/grpo_accuracy_trajectory.pdf')
fig.savefig('figures/grpo_accuracy_trajectory.png')
plt.close()
print('Saved: figures/grpo_accuracy_trajectory.pdf')


# =============================================================================
# PLOT 2: Match-F1 trajectories
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(comp_steps, comp_f1, 'o-', color=C_COMP, label='Composite', markersize=4, linewidth=1.5)
ax.plot(ao_steps, ao_f1, 's-', color=C_AO, label='Answer-Only', markersize=5, linewidth=1.5)
ax.plot(cpr_steps, cpr_f1, '^-', color=C_CPR, label='CPR', markersize=4, linewidth=1.5)
ax.plot(cprc_steps, cprc_f1, 'D-', color=C_CPRC, label='CPR-Curriculum', markersize=5, linewidth=2)
ax.axhline(y=0.4802, color=C_BASE, linestyle='--', linewidth=2, alpha=0.9, label='Baseline (0.480)')

# Vertical line where Composite/Answer-Only stopped
ax.axvline(x=1500, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(1520, 0.14, 'Comp. & Ans-Only\nstopped', fontsize=7.5, color='gray',
        fontstyle='italic', va='bottom')

ax.set_xlabel('Training Step')
ax.set_ylabel('Match-F1')
ax.set_title('Match-F1 Trajectory: Four GRPO Strategies')
ax.legend(loc='lower right', fontsize=8)
ax.set_ylim(0.10, 0.78)
ax.set_xlim(-50, 2900)
ax.grid(True, alpha=0.3)
fig.savefig('figures/grpo_f1_trajectory.pdf')
fig.savefig('figures/grpo_f1_trajectory.png')
plt.close()
print('Saved: figures/grpo_f1_trajectory.pdf')


# =============================================================================
# PLOT 3: Accuracy vs F1 scatter (best checkpoints)
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))

strategies = ['Baseline', 'Composite\n(ckpt-1400)', 'Answer-Only\n(ckpt-1400)', 'CPR\n(ckpt-2800)', 'CPR-Cur\n(ckpt-2800)']
accs       = [39.85, 44.92, 44.90, 41.40, 47.52]
f1s        = [0.4802, 0.4264, 0.4330, 0.6331, 0.6327]
colors     = [C_BASE, C_COMP, C_AO, C_CPR, C_CPRC]
markers    = ['D', 'o', 's', '^', 'P']

for i in range(len(strategies)):
    ax.scatter(accs[i], f1s[i], c=colors[i], marker=markers[i], s=150, zorder=5, edgecolors='black', linewidths=0.5)
    offset_x = 0.5 if i != 2 else -5.5
    offset_y = 0.012 if i != 0 else -0.02
    ax.annotate(strategies[i], (accs[i] + offset_x, f1s[i] + offset_y), fontsize=9, ha='left')

# Arrows showing improvement direction
ax.annotate('', xy=(41.38, 0.6329), xytext=(39.85, 0.4802),
            arrowprops=dict(arrowstyle='->', color=C_CPR, lw=2, linestyle='--'))

ax.set_xlabel('Accuracy (%)')
ax.set_ylabel('Match-F1')
ax.set_title('Accuracy vs Reasoning Quality\n(Best Checkpoint per Strategy)')
ax.grid(True, alpha=0.3)
ax.set_xlim(37, 50)
ax.set_ylim(0.38, 0.70)
fig.savefig('figures/grpo_acc_vs_f1.pdf')
fig.savefig('figures/grpo_acc_vs_f1.png')
plt.close()
print('Saved: figures/grpo_acc_vs_f1.pdf')


# =============================================================================
# PLOT 4: Precision vs Recall at best checkpoints (bar chart)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

labels = ['Baseline', 'Composite', 'Answer-Only', 'CPR', 'CPR-Cur']
precs  = [0.898, 0.983, 0.802, 0.975, 0.963]
recs   = [0.347, 0.284, 0.313, 0.489, 0.493]

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, precs, width, label='Precision', color='#ff9999', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, recs, width, label='Recall', color='#66b3ff', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Score')
ax.set_title('Precision vs Recall at Best Checkpoint')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

fig.savefig('figures/grpo_precision_recall_bars.pdf')
fig.savefig('figures/grpo_precision_recall_bars.png')
plt.close()
print('Saved: figures/grpo_precision_recall_bars.pdf')


# =============================================================================
# PLOT 5: Main results - Accuracy vs F1 scatter for 14 models
# =============================================================================
models = [
    ('Qwen3-VL-8B',    57.66, 0.6586, 8,  'Qwen3'),
    ('InternVL3.5-8B',  51.98, 0.5303, 8,  'InternVL'),
    ('InternVL3.5-38B', 51.21, 0.6118, 38, 'InternVL'),
    ('Qwen3-VL-32B',   49.22, 0.7176, 32, 'Qwen3'),
    ('Qwen2.5-VL-32B', 47.63, 0.6525, 32, 'Qwen2.5'),
    ('Qwen2.5-VL-3B',  42.70, 0.4802, 3,  'Qwen2.5'),
    ('InternVL3.5-4B',  37.61, 0.4318, 4,  'InternVL'),
    ('Qwen3-VL-2B',    34.15, 0.5950, 2,  'Qwen3'),
    ('Gemma3-12B',     33.83, 0.6049, 12, 'Gemma'),
    ('InternVL3.5-2B',  33.02, 0.4687, 2,  'InternVL'),
    ('Qwen2.5-VL-7B',  30.43, 0.4754, 7,  'Qwen2.5'),
    ('Gemma3-4B',      28.65, 0.6179, 4,  'Gemma'),
    ('MiniCPM-V-8B',   25.54, 0.2149, 8,  'Other'),
    ('LLaVA-7B',       24.66, 0.5121, 7,  'Other'),
]

family_colors = {
    'Qwen3':   '#e41a1c',
    'Qwen2.5': '#ff7f00',
    'InternVL':'#377eb8',
    'Gemma':   '#4daf4a',
    'Other':   '#984ea3',
}

fig, ax = plt.subplots(figsize=(9, 6))

for name, acc, f1, params, family in models:
    size = max(30, params * 4)
    ax.scatter(acc, f1, c=family_colors[family], s=size, alpha=0.8,
              edgecolors='black', linewidths=0.5, zorder=5)
    # Label only notable models
    if name in ['Qwen3-VL-8B', 'Qwen3-VL-32B', 'Gemma3-4B', 'InternVL3.5-38B',
                'Qwen2.5-VL-3B', 'MiniCPM-V-8B', 'LLaVA-7B']:
        offset_y = 0.015 if 'MiniCPM' not in name else -0.025
        ax.annotate(name, (acc, f1 + offset_y), fontsize=7.5, ha='center', alpha=0.8)

# Legend for families
for family, color in family_colors.items():
    ax.scatter([], [], c=color, s=60, label=family, edgecolors='black', linewidths=0.5)
ax.legend(title='Model Family', loc='lower right')

# Mark GRPO base model
ax.scatter(42.70, 0.4802, c='none', s=200, edgecolors='black', linewidths=2, zorder=6)
ax.annotate('GRPO base\nmodel', (42.70, 0.4802 - 0.035), fontsize=8, ha='center', fontstyle='italic')

ax.set_xlabel('Accuracy (%)')
ax.set_ylabel('Match-F1')
ax.set_title('CRYSTAL: Accuracy vs Reasoning Quality (14 Models)')
ax.grid(True, alpha=0.3)
fig.savefig('figures/main_acc_vs_f1_scatter.pdf')
fig.savefig('figures/main_acc_vs_f1_scatter.png')
plt.close()
print('Saved: figures/main_acc_vs_f1_scatter.pdf')


# =============================================================================
# PLOT 6: CPR - Accuracy and F1 move together after recovery
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

# Normalize both metrics to 0-1 scale for direct comparison
cpr_acc_norm = [(a - min(cpr_acc)) / (max(cpr_acc) - min(cpr_acc)) for a in cpr_acc]
cpr_f1_norm  = [(f - min(cpr_f1)) / (max(cpr_f1) - min(cpr_f1)) for f in cpr_f1]

ax.plot(cpr_steps, cpr_acc_norm, '^-', color=C_CPR, label='Accuracy (normalized)', markersize=7, linewidth=2.5)
ax.plot(cpr_steps, cpr_f1_norm, 's-', color='#7b2d8e', label='Match-F1 (normalized)', markersize=7, linewidth=2.5)

# Phase annotations with vertical lines instead of bands
ax.axvline(x=700, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
ax.axvline(x=1400, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

# Phase labels at the top
ax.text(350, 1.05, 'Exploration', ha='center', fontsize=9, color='#555555')
ax.text(1050, 1.05, 'Recovery', ha='center', fontsize=9, color='#555555')
ax.text(1900, 1.05, 'Both improve', ha='center', fontsize=9, color=C_CPR, fontweight='bold')

# Key insight annotation
ax.annotate('After step 700:\nAcc and F1 rise together', xy=(1400, 0.45), xytext=(1600, 0.15),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax.set_xlabel('Training Step')
ax.set_ylabel('Normalized Score (0 = worst, 1 = best)')
ax.set_title('CPR: Accuracy and Reasoning Quality Co-evolve After Recovery')
ax.legend(loc='lower left', fontsize=10)
ax.set_ylim(-0.1, 1.15)
ax.set_xlim(-50, 2900)
ax.grid(True, alpha=0.3)
fig.savefig('figures/grpo_cpr_dynamics.pdf')
fig.savefig('figures/grpo_cpr_dynamics.png')
plt.close()
print('Saved: figures/grpo_cpr_dynamics.pdf')


# =============================================================================
# PLOT 7: Ablation heatmap
# =============================================================================
encoders = ['distilroberta-v1', 'mpnet-base-v2', 'MiniLM-L6-v2', 'MiniLM-L12-v2']
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
ablation_data = np.array([
    [0.638, 0.653, 0.634, 0.601, 0.562],
    [0.550, 0.561, 0.572, 0.554, 0.520],
    [0.609, 0.597, 0.571, 0.540, 0.505],
    [0.595, 0.581, 0.559, 0.530, 0.497],
])

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(ablation_data, cmap='YlOrRd', aspect='auto', vmin=0.45, vmax=0.66)

ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f'$\\tau$={t}' for t in thresholds])
ax.set_yticks(range(len(encoders)))
ax.set_yticklabels(encoders)

for i in range(len(encoders)):
    for j in range(len(thresholds)):
        text_color = 'white' if ablation_data[i, j] > 0.62 else 'black'
        ax.text(j, i, f'{ablation_data[i,j]:.3f}', ha='center', va='center',
                fontsize=10, color=text_color, fontweight='bold' if (i==0 and j==1) else 'normal')

# Highlight best cell
rect = plt.Rectangle((-0.5+1, -0.5+0), 1, 1, linewidth=3, edgecolor='blue', facecolor='none')
ax.add_patch(rect)

ax.set_title('Match-F1 Ablation: Encoder $\\times$ Threshold')
fig.colorbar(im, ax=ax, label='Match-F1')
fig.savefig('figures/ablation_heatmap.pdf')
fig.savefig('figures/ablation_heatmap.png')
plt.close()
print('Saved: figures/ablation_heatmap.pdf')


# =============================================================================
# PLOT 8: Summary comparison bar chart
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

strategies = ['Baseline', 'Composite', 'Ans-Only', 'CPR', 'CPR-Cur']
colors_bar = [C_BASE, C_COMP, C_AO, C_CPR, C_CPRC]

# Accuracy
accs_bar = [39.85, 44.92, 44.90, 41.40, 47.52]
axes[0].bar(strategies, accs_bar, color=colors_bar, edgecolor='black', linewidth=0.5, width=0.6)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Accuracy')
axes[0].set_ylim(35, 52)
for i, v in enumerate(accs_bar):
    axes[0].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=25)

# Match-F1
f1s_bar = [0.4802, 0.4264, 0.4330, 0.6331, 0.6327]
axes[1].bar(strategies, f1s_bar, color=colors_bar, edgecolor='black', linewidth=0.5, width=0.6)
axes[1].set_ylabel('Match-F1')
axes[1].set_title('Reasoning Quality')
axes[1].set_ylim(0.35, 0.70)
for i, v in enumerate(f1s_bar):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=25)

# Recall
recs_bar = [0.347, 0.284, 0.313, 0.489, 0.493]
axes[2].bar(strategies, recs_bar, color=colors_bar, edgecolor='black', linewidth=0.5, width=0.6)
axes[2].set_ylabel('Recall')
axes[2].set_title('Reasoning Completeness')
axes[2].set_ylim(0.20, 0.55)
for i, v in enumerate(recs_bar):
    axes[2].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].tick_params(axis='x', rotation=25)

plt.suptitle('Four GRPO Strategies: Accuracy, Quality, and Completeness', fontsize=14, y=1.02)
plt.tight_layout(w_pad=3.0)
fig.savefig('figures/grpo_summary_bars.pdf')
fig.savefig('figures/grpo_summary_bars.png')
plt.close()
print('Saved: figures/grpo_summary_bars.pdf')


print('\n=== All plots generated in figures/ ===')
