#!/usr/bin/env python3
"""
Find Wrong+Sound examples with F1 around 0.85 but Precision < 1.0
This shows the model generates SOME incorrect steps (not just omissions)
"""

import pandas as pd
from pathlib import Path

# Models to check
models = {
    'outputs_testing_gemma3_4b': 'Gemma3-4B',
    'outputs_testing_gemma3_12b_64k': 'Gemma3-12B',
    'outputs_testing_llava7b_16': 'LLaVA-v1.6-7B',
    'outputs_testing_minicpm_v_8b': 'MiniCPM-v2.6-8B',
    'outputs_testing_internvl35_2b': 'InternVL3.5-2B',
    'outputs_testing_internvl35_4b': 'InternVL3.5-4B',
    'outputs_testing_qwen3vl_32b': 'Qwen3-VL-32B',
}

base_path = Path("/gpudata3/Wayner/VLM-R1/final_table")

all_candidates = []

for model_dir, model_name in models.items():
    csv_path = base_path / model_dir / "metrics_detailed.csv"

    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)

    # Filter: Wrong answer, F1 0.8-0.9, Precision < 1.0 (not perfect!), short steps
    candidates = df[
        (df['accuracy_correct'] == False) &
        (df['match_f1'] >= 0.80) &
        (df['match_f1'] <= 0.90) &
        (df['precision'] < 1.0) &  # KEY: Precision NOT perfect
        (df['precision'] >= 0.70) &  # But still reasonably high
        (df['num_predicted_steps'] <= 10)
    ].copy()

    candidates['model'] = model_name
    candidates['f1_distance_from_85'] = abs(candidates['match_f1'] - 0.85)

    all_candidates.append(candidates)

    print(f"{model_name}: {len(candidates)} candidates (Precision < 1.0)")

# Combine
all_df = pd.concat(all_candidates, ignore_index=True)
all_df = all_df.sort_values(['f1_distance_from_85', 'num_predicted_steps'],
                             ascending=[True, True])

print(f"\n{'='*100}")
print(f"TOTAL: {len(all_df)} candidates with Precision < 1.0")
print(f"{'='*100}\n")

# Show top 30
print("TOP 30 CANDIDATES (F1~0.85, Precision < 1.0 showing actual errors)")
print("="*100)
print(all_df.head(30)[['model', 'sample_idx', 'match_f1', 'precision', 'recall',
                        'num_predicted_steps', 'num_reference_steps', 'answer']].to_string(index=False))

# Save
import json
output_path = "/gpudata3/Wayner/VLM-R1/wrongsound_precision_imperfect.json"
with open(output_path, 'w') as f:
    json.dump({
        'total_candidates': len(all_df),
        'top_30': all_df.head(30).to_dict('records')
    }, f, indent=2)

print(f"\n\nSaved to: {output_path}")
print("\nRECOMMENDED: Pick one with:")
print("  - F1 closest to 0.85")
print("  - Precision in 0.8-0.9 range (shows some incorrect steps)")
print("  - Fewest steps (≤8 ideal)")
print("  - Gemma3-4B or other non-Qwen model for diversity")
