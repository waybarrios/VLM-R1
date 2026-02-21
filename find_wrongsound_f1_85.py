#!/usr/bin/env python3
"""
Find Wrong+Sound examples with F1 around 0.85
Criteria:
- Accuracy = False (wrong answer)
- 0.80 <= F1 <= 0.90 (high but not perfect)
- num_predicted_steps <= 10 (SHORT - to fit in page)
"""

import pandas as pd
import json

# Load metrics
csv_path = "/gpudata3/Wayner/VLM-R1/final_table/outputs_testing_qwen3vl_32b/metrics_detailed.csv"
df = pd.read_csv(csv_path)

print(f"Total samples: {len(df)}")

# Filter for Wrong+Sound with F1 around 0.85
candidates = df[
    (df['accuracy_correct'] == False) &  # Wrong answer
    (df['match_f1'] >= 0.80) &             # High F1
    (df['match_f1'] <= 0.90) &             # But not perfect (0.8-0.9)
    (df['num_predicted_steps'] <= 10)     # SHORT (10 steps or less)
].copy()

print(f"Found {len(candidates)} candidates with 0.80 <= F1 <= 0.90 and ≤10 steps\n")

if len(candidates) == 0:
    # Try relaxing to ≤12 steps
    candidates = df[
        (df['accuracy_correct'] == False) &
        (df['match_f1'] >= 0.80) &
        (df['match_f1'] <= 0.90) &
        (df['num_predicted_steps'] <= 12)
    ].copy()
    print(f"Relaxed to ≤12 steps: {len(candidates)} candidates\n")

# Sort by F1 (prefer around 0.85), then by num_predicted_steps (ascending = prefer shorter)
candidates['f1_distance_from_85'] = abs(candidates['match_f1'] - 0.85)
candidates = candidates.sort_values(['f1_distance_from_85', 'num_predicted_steps'],
                                     ascending=[True, True])

# Show top 20 candidates
print("=" * 100)
print("TOP 20 CANDIDATES (F1 closest to 0.85, sorted by steps)")
print("=" * 100)
print(candidates.head(20)[['sample_idx', 'accuracy_correct', 'match_f1', 'precision',
                            'recall', 'num_predicted_steps', 'num_reference_steps',
                            'answer']].to_string(index=False))

# Save top candidates
output = {
    'total_candidates': len(candidates),
    'top_20': candidates.head(20).to_dict('records')
}

output_path = "/gpudata3/Wayner/VLM-R1/wrongsound_candidates_f1_85.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\nSaved candidates to: {output_path}")
print("\nRECOMMENDED: Pick sample_idx with:")
print("  - F1 closest to 0.85")
print("  - Fewest predicted steps (≤8 ideal)")
print("  - Interesting visual problem")
