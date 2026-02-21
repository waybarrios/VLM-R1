#!/usr/bin/env python3
"""
Find Wrong+Sound examples with wider F1 range for InternVL models
"""

import pandas as pd
from pathlib import Path

# InternVL models
models = {
    'outputs_testing_internvl35_1b': 'InternVL3.5-1B',
    'outputs_testing_internvl35_2b': 'InternVL3.5-2B',
    'outputs_testing_internvl35_4b': 'InternVL3.5-4B',
    'outputs_testing_internvl35_8b': 'InternVL3.5-8B',
    'outputs_testing_internvl35_38b': 'InternVL3.5-38B',
}

base_path = Path("/gpudata3/Wayner/VLM-R1/final_table")

for model_dir, model_name in models.items():
    csv_path = base_path / model_dir / "metrics_detailed.csv"

    if not csv_path.exists():
        print(f"⚠️  Skipping {model_name} (file not found)")
        continue

    df = pd.read_csv(csv_path)

    # Wider range for InternVL
    candidates = df[
        (df['accuracy_correct'] == False) &
        (df['match_f1'] >= 0.75) &  # Wider: 0.75-0.95
        (df['match_f1'] <= 0.95) &
        (df['num_predicted_steps'] <= 10)
    ].copy()

    candidates['f1_distance_from_85'] = abs(candidates['match_f1'] - 0.85)
    candidates = candidates.sort_values('f1_distance_from_85')

    print(f"\n{'='*80}")
    print(f"{model_name}: {len(candidates)} candidates (F1 0.75-0.95)")
    print('='*80)

    if len(candidates) > 0:
        print("\nTop 10:")
        print(candidates.head(10)[['sample_idx', 'match_f1', 'precision', 'recall',
                                    'num_predicted_steps', 'num_reference_steps',
                                    'answer']].to_string(index=False))
