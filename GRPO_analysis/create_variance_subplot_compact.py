#!/usr/bin/env python3
"""
Create GRPO training figure with variance subplot - COMPACT HORIZONTAL LAYOUT
5 subplots in a single row to save space
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('consolidated_summary.json') as f:
    data = json.load(f)

# Extract data
results = data['results']
steps = []
accuracy = []
match_f1 = []
f1_std = []
precision = []
recall = []

baseline_idx = None
for i, result in enumerate(results):
    if result['step'] == -1:
        baseline_idx = i
        baseline_accuracy = result['accuracy']
        baseline_f1 = result['match_f1_mean']
        baseline_f1_std = result['match_f1_std']
        baseline_precision = result['precision_mean']
        baseline_recall = result['recall_mean']
    else:
        steps.append(result['step'])
        accuracy.append(result['accuracy'])
        match_f1.append(result['match_f1_mean'])
        f1_std.append(result['match_f1_std'])
        precision.append(result['precision_mean'])
        recall.append(result['recall_mean'])

steps = np.array(steps)
accuracy = np.array(accuracy)
match_f1 = np.array(match_f1)
f1_std = np.array(f1_std)
precision = np.array(precision)
recall = np.array(recall)

# Create figure with 5 subplots in a single row (1x5)
fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))

# Style parameters
linewidth = 2
markersize = 5
baseline_color = '#d62728'  # Red
grpo_color = '#1f77b4'      # Blue
alpha = 0.8

# (a) Accuracy
axes[0].plot(steps, accuracy * 100, 'o-', color=grpo_color, linewidth=linewidth,
             markersize=markersize, alpha=alpha)
axes[0].axhline(y=baseline_accuracy * 100, color=baseline_color, linestyle='--',
                linewidth=linewidth, alpha=alpha)
axes[0].set_xlabel('Training Step', fontsize=10)
axes[0].set_ylabel('Accuracy (%)', fontsize=10)
axes[0].set_title('(a) Accuracy', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# (b) Match F1
axes[1].plot(steps, match_f1, 'o-', color=grpo_color, linewidth=linewidth,
             markersize=markersize, alpha=alpha)
axes[1].axhline(y=baseline_f1, color=baseline_color, linestyle='--',
                linewidth=linewidth, alpha=alpha)
axes[1].set_xlabel('Training Step', fontsize=10)
axes[1].set_ylabel('Match F1', fontsize=10)
axes[1].set_title('(b) Match F1', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# (c) Match F1 Standard Deviation - NO ANNOTATION
axes[2].plot(steps, f1_std, 'o-', color=grpo_color, linewidth=linewidth,
             markersize=markersize, alpha=alpha)
axes[2].axhline(y=baseline_f1_std, color=baseline_color, linestyle='--',
                linewidth=linewidth, alpha=alpha)
axes[2].set_xlabel('Training Step', fontsize=10)
axes[2].set_ylabel('Std Dev', fontsize=10)
axes[2].set_title('(c) Variance', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)

# (d) Precision
axes[3].plot(steps, precision, 'o-', color=grpo_color, linewidth=linewidth,
             markersize=markersize, alpha=alpha)
axes[3].axhline(y=baseline_precision, color=baseline_color, linestyle='--',
                linewidth=linewidth, alpha=alpha)
axes[3].set_xlabel('Training Step', fontsize=10)
axes[3].set_ylabel('Precision', fontsize=10)
axes[3].set_title('(d) Precision', fontsize=11, fontweight='bold')
axes[3].grid(True, alpha=0.3)

# (e) Recall
axes[4].plot(steps, recall, 'o-', color=grpo_color, linewidth=linewidth,
             markersize=markersize, alpha=alpha)
axes[4].axhline(y=baseline_recall, color=baseline_color, linestyle='--',
                linewidth=linewidth, alpha=alpha)
axes[4].set_xlabel('Training Step', fontsize=10)
axes[4].set_ylabel('Recall', fontsize=10)
axes[4].set_title('(e) Recall', fontsize=11, fontweight='bold')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
plt.savefig('grpo_training_compact_5plots.pdf', bbox_inches='tight', dpi=300)
plt.savefig('grpo_training_compact_5plots.png', bbox_inches='tight', dpi=300)
print("✅ Created: grpo_training_compact_5plots.pdf")
print("✅ Created: grpo_training_compact_5plots.png")

plt.close('all')

# Print summary statistics
print("\n📊 Variance Reduction Summary:")
print(f"   Baseline Std Dev: ±{baseline_f1_std:.4f}")
print(f"   Min GRPO Std Dev: ±{f1_std.min():.4f} (step {steps[f1_std.argmin()]})")
print(f"   Reduction Factor: {baseline_f1_std / f1_std.min():.2f}×")
