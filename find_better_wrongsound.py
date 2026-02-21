#!/usr/bin/env python3
"""
Find better Wrong+Sound examples with FEWER reasoning steps
Criteria:
- Accuracy = False (wrong answer)
- F1 >= 0.7 (high quality reasoning)
- Precision >= 0.7
- Recall >= 0.7
- num_predicted_steps <= 10 (SHORT - to fit in page)
"""

import pandas as pd
import json

# Load metrics
csv_path = "/gpudata3/Wayner/VLM-R1/final_table/outputs_testing_qwen3vl_32b/metrics_detailed.csv"
df = pd.read_csv(csv_path)

print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

# Filter for Wrong+Sound with SHORT reasoning steps
candidates = df[
    (df['accuracy_correct'] == False) &  # Wrong answer
    (df['match_f1'] >= 0.7) &             # High F1
    (df['precision'] >= 0.7) &            # High precision
    (df['recall'] >= 0.7) &               # High recall
    (df['num_predicted_steps'] <= 10)     # SHORT (10 steps or less)
].copy()

print(f"Found {len(candidates)} candidates with ≤10 steps\n")

if len(candidates) == 0:
    # Try relaxing to ≤12 steps
    candidates = df[
        (df['accuracy_correct'] == False) &
        (df['match_f1'] >= 0.7) &
        (df['precision'] >= 0.7) &
        (df['recall'] >= 0.7) &
        (df['num_predicted_steps'] <= 12)
    ].copy()
    print(f"Relaxed to ≤12 steps: {len(candidates)} candidates\n")

# Sort by F1 (descending) then by num_predicted_steps (ascending = prefer shorter)
candidates = candidates.sort_values(['match_f1', 'num_predicted_steps'],
                                     ascending=[False, True])

# Show top 20 candidates
print("=" * 100)
print("TOP 20 CANDIDATES (sorted by F1 desc, then steps asc)")
print("=" * 100)
print(candidates.head(20)[['sample_idx', 'accuracy_correct', 'match_f1', 'precision',
                            'recall', 'num_predicted_steps', 'num_reference_steps',
                            'answer']].to_string(index=False))

# Save top candidates
output = {
    'total_candidates': len(candidates),
    'top_20': candidates.head(20).to_dict('records')
}

output_path = "/gpudata3/Wayner/VLM-R1/wrongsound_candidates_short.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\nSaved candidates to: {output_path}")
print("\nRECOMMENDED: Pick sample_idx with:")
print("  - Highest F1 (≥0.8 preferred)")
print("  - Fewest predicted steps (≤8 ideal)")
print("  - Interesting visual problem (check image)")
