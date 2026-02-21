#!/usr/bin/env python3
"""
Find Wrong+Sound examples with F1 around 0.85 across different models
"""

import pandas as pd
import json
from pathlib import Path

# Models to check
models = {
    'outputs_testing_gemma3_4b': 'Gemma3-4B',
    'outputs_testing_gemma3_12b_64k': 'Gemma3-12B',
    'outputs_testing_llava7b_16': 'LLaVA-v1.6-7B',
    'outputs_testing_minicpm_v_8b': 'MiniCPM-v2.6-8B',
    'outputs_testing_internvl35_8b': 'InternVL3.5-8B',
}

base_path = Path("/gpudata3/Wayner/VLM-R1/final_table")

all_candidates = []

for model_dir, model_name in models.items():
    csv_path = base_path / model_dir / "metrics_detailed.csv"

    if not csv_path.exists():
        print(f"⚠️  Skipping {model_name} (file not found)")
        continue

    df = pd.read_csv(csv_path)

    # Filter for Wrong+Sound with F1 around 0.85
    candidates = df[
        (df['accuracy_correct'] == False) &
        (df['match_f1'] >= 0.80) &
        (df['match_f1'] <= 0.90) &
        (df['num_predicted_steps'] <= 10)
    ].copy()

    candidates['model'] = model_name
    candidates['f1_distance_from_85'] = abs(candidates['match_f1'] - 0.85)

    all_candidates.append(candidates)

    print(f"{model_name}: {len(candidates)} candidates")

# Combine all
all_df = pd.concat(all_candidates, ignore_index=True)
all_df = all_df.sort_values(['f1_distance_from_85', 'num_predicted_steps'],
                             ascending=[True, True])

print(f"\n{'='*100}")
print(f"TOTAL: {len(all_df)} candidates across all models")
print(f"{'='*100}\n")

# Show top 30
print("TOP 30 CANDIDATES (F1 closest to 0.85, all models)")
print("="*100)
print(all_df.head(30)[['model', 'sample_idx', 'match_f1', 'precision', 'recall',
                        'num_predicted_steps', 'num_reference_steps', 'answer']].to_string(index=False))

# Save
output_path = "/gpudata3/Wayner/VLM-R1/wrongsound_multi_model_f1_85.json"
with open(output_path, 'w') as f:
    json.dump({
        'total_candidates': len(all_df),
        'top_30': all_df.head(30).to_dict('records')
    }, f, indent=2)

print(f"\n\nSaved to: {output_path}")
