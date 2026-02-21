#!/usr/bin/env python3
"""
Find Wrong+Sound examples NOT from ScienceQA
With F1~0.85, Precision < 1.0, short steps
"""

import pandas as pd
from pathlib import Path
import pyarrow as pa
import glob

# Load dataset to get source info
DATASET_PATH = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
arrow_files = sorted(glob.glob(f"{DATASET_PATH}/data-*.arrow"))
tables = []
for arrow_file in arrow_files:
    with pa.memory_map(arrow_file, 'r') as source:
        reader = pa.ipc.open_stream(source)
        tables.append(reader.read_all())

full_table = pa.concat_tables(tables)

# Extract source column
sources = {}
for i in range(len(full_table)):
    row = full_table.slice(i, 1)
    source = row.column('source')[0].as_py()
    sources[i] = source

# Models to check
models = {
    'outputs_testing_gemma3_4b': 'Gemma3-4B',
    'outputs_testing_gemma3_12b_64k': 'Gemma3-12B',
    'outputs_testing_internvl35_2b': 'InternVL3.5-2B',
    'outputs_testing_internvl35_4b': 'InternVL3.5-4B',
    'outputs_testing_llava7b_16': 'LLaVA-v1.6-7B',
    'outputs_testing_qwen3vl_32b': 'Qwen3-VL-32B',
}

base_path = Path("/gpudata3/Wayner/VLM-R1/final_table")

all_candidates = []

for model_dir, model_name in models.items():
    csv_path = base_path / model_dir / "metrics_detailed.csv"

    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)

    # Filter: Wrong, F1 0.8-0.9, Precision < 1.0, short
    candidates = df[
        (df['accuracy_correct'] == False) &
        (df['match_f1'] >= 0.80) &
        (df['match_f1'] <= 0.90) &
        (df['precision'] < 1.0) &
        (df['precision'] >= 0.70) &
        (df['num_predicted_steps'] <= 10)
    ].copy()

    # Add source info
    candidates['source'] = candidates['sample_idx'].map(sources)

    # Filter OUT ScienceQA
    candidates = candidates[candidates['source'] != 'ScienceQA'].copy()

    candidates['model'] = model_name
    candidates['f1_distance_from_85'] = abs(candidates['match_f1'] - 0.85)

    all_candidates.append(candidates)

    print(f"{model_name}: {len(candidates)} non-ScienceQA candidates")

# Combine
all_df = pd.concat(all_candidates, ignore_index=True)
all_df = all_df.sort_values(['f1_distance_from_85', 'num_predicted_steps'],
                             ascending=[True, True])

print(f"\n{'='*100}")
print(f"TOTAL: {len(all_df)} NON-SCIENCEQA candidates")
print(f"{'='*100}\n")

# Show top 30 with source
print("TOP 30 NON-SCIENCEQA CANDIDATES")
print("="*100)
print(all_df.head(30)[['model', 'sample_idx', 'source', 'match_f1', 'precision', 'recall',
                        'num_predicted_steps', 'num_reference_steps']].to_string(index=False))

# Save
import json
output_path = "/gpudata3/Wayner/VLM-R1/wrongsound_not_scienceqa.json"
with open(output_path, 'w') as f:
    json.dump({
        'total_candidates': len(all_df),
        'top_30': all_df.head(30).to_dict('records')
    }, f, indent=2)

print(f"\n\nSaved to: {output_path}")

# Show source distribution
print("\n" + "="*100)
print("SOURCE DISTRIBUTION")
print("="*100)
print(all_df['source'].value_counts())
