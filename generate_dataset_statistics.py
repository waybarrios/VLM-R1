#!/usr/bin/env python3
"""
Generate CRYSTAL Dataset Statistics for Paper
Analyzes the CRYSTAL benchmark to produce comprehensive statistics
for inclusion in academic publications.

Output: dataset_source_statistics.json
"""

import pyarrow as pa
import json
from collections import defaultdict
import numpy as np
from pathlib import Path

def load_complexity_scores(complexity_file: str = "dataset_complexity_scores.json"):
    """Load pre-computed complexity scores"""
    with open(complexity_file) as f:
        return json.load(f)

def analyze_dataset(dataset_path: str, complexity_data: dict):
    """Analyze dataset composition by source"""

    # Create complexity map
    complexity_map = {
        s['sample_idx']: {
            'complexity_level': s['complexity_level'],
            'num_reference_steps': s['features']['num_reference_steps']
        }
        for s in complexity_data['samples']
    }

    # Arrow files
    arrow_files = [
        f"{dataset_path}/data-00000-of-00004.arrow",
        f"{dataset_path}/data-00001-of-00004.arrow",
        f"{dataset_path}/data-00002-of-00004.arrow",
        f"{dataset_path}/data-00003-of-00004.arrow"
    ]

    # Initialize statistics
    source_stats = defaultdict(lambda: {
        'count': 0,
        'steps': [],
        'easy': 0,
        'medium': 0,
        'hard': 0,
        'very_hard': 0
    })

    total_samples = 0
    all_steps = []

    # Read and analyze each arrow file
    for arrow_file in arrow_files:
        with pa.memory_map(arrow_file, 'r') as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()

            # Get columns
            sources = table.column('source').to_pylist()
            ref_steps = table.column('reference_steps').to_pylist()

            for i in range(len(sources)):
                source = sources[i] if sources[i] else 'unknown'
                steps = ref_steps[i] if ref_steps[i] else []
                num_steps = len(steps)

                source_stats[source]['count'] += 1
                source_stats[source]['steps'].append(num_steps)
                all_steps.append(num_steps)

                # Get complexity level
                if total_samples in complexity_map:
                    level = complexity_map[total_samples]['complexity_level']
                    source_stats[source][level] += 1

                total_samples += 1

    return source_stats, total_samples, all_steps

def generate_statistics_json(source_stats, total_samples, all_steps, complexity_data):
    """Generate JSON output with statistics"""

    output_data = {
        'metadata': {
            'dataset_name': 'CRYSTAL',
            'total_samples': total_samples,
            'num_sources': len(source_stats),
            'avg_steps_overall': float(np.mean(all_steps)),
            'median_steps_overall': float(np.median(all_steps)),
            'std_steps_overall': float(np.std(all_steps)),
            'min_steps': int(np.min(all_steps)),
            'max_steps': int(np.max(all_steps))
        },
        'complexity_distribution': complexity_data['level_counts'],
        'sources': {}
    }

    # Add source-specific statistics
    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        output_data['sources'][source] = {
            'count': stats['count'],
            'percentage': round(100 * stats['count'] / total_samples, 1),
            'avg_steps': round(float(np.mean(stats['steps'])), 1),
            'median_steps': round(float(np.median(stats['steps'])), 1),
            'std_steps': round(float(np.std(stats['steps'])), 1),
            'min_steps': int(np.min(stats['steps'])),
            'max_steps': int(np.max(stats['steps'])),
            'complexity': {
                'easy': stats['easy'],
                'medium': stats['medium'],
                'hard': stats['hard'],
                'very_hard': stats['very_hard']
            }
        }

    return output_data

def print_statistics_table(source_stats, total_samples, all_steps, complexity_data):
    """Print formatted statistics table"""

    print("=" * 100)
    print("CRYSTAL BENCHMARK DATASET STATISTICS")
    print("=" * 100)
    print(f"\nTotal Samples: {total_samples:,}")
    print(f"Total Sources: {len(source_stats)}")
    print(f"Average Steps: {np.mean(all_steps):.1f}")
    print(f"Median Steps: {np.median(all_steps):.1f}")
    print(f"Std Steps: {np.std(all_steps):.1f}")

    print("\n" + "=" * 100)
    print(f"{'Source':<30} {'Count':>8} {'%':>7} {'Avg Steps':>11} {'Easy':>7} {'Med':>7} {'Hard':>7}")
    print("=" * 100)

    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        count = stats['count']
        pct = 100 * count / total_samples
        avg_steps = np.mean(stats['steps']) if stats['steps'] else 0

        print(f"{source:<30} {count:>8,} {pct:>6.1f}% {avg_steps:>10.1f}  "
              f"{stats['easy']:>6,} {stats['medium']:>6,} {stats['hard']:>6,}")

    print("=" * 100)
    print(f"{'TOTAL':<30} {total_samples:>8,} {100.0:>6.1f}% {np.mean(all_steps):>10.1f}  "
          f"{complexity_data['level_counts']['easy']:>6,} "
          f"{complexity_data['level_counts']['medium']:>6,} "
          f"{complexity_data['level_counts']['hard']:>6,}")
    print("=" * 100)

def main():
    """Main execution"""

    # Configuration
    dataset_path = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
    complexity_file = "dataset_complexity_scores.json"
    output_file = "dataset_source_statistics.json"

    print("📊 Generating CRYSTAL Dataset Statistics")
    print(f"Dataset: {dataset_path}")
    print()

    # Load complexity scores
    print("📂 Loading complexity scores...")
    complexity_data = load_complexity_scores(complexity_file)
    print(f"✓ Loaded complexity for {complexity_data['metadata']['total_samples']} samples")
    print()

    # Analyze dataset
    print("📊 Analyzing dataset composition...")
    source_stats, total_samples, all_steps = analyze_dataset(dataset_path, complexity_data)
    print(f"✓ Analyzed {total_samples:,} samples from {len(source_stats)} sources")
    print()

    # Print statistics
    print_statistics_table(source_stats, total_samples, all_steps, complexity_data)
    print()

    # Generate JSON
    output_data = generate_statistics_json(source_stats, total_samples, all_steps, complexity_data)

    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Statistics saved to: {output_file}")
    print()

    # Summary for paper
    print("=" * 100)
    print("SUMMARY FOR PAPER")
    print("=" * 100)
    print("\nDataset Composition:")
    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        pct = 100 * stats['count'] / total_samples
        avg_steps = np.mean(stats['steps'])
        print(f"  • {source:20s}: {stats['count']:>5,} samples ({pct:>5.1f}%), {avg_steps:>5.1f} avg steps")

    print("\nComplexity Distribution:")
    for level in ['easy', 'medium', 'hard', 'very_hard']:
        count = complexity_data['level_counts'][level]
        pct = 100 * count / total_samples
        print(f"  • {level.capitalize():12s}: {count:>5,} samples ({pct:>5.1f}%)")

    print("\nKey Statistics:")
    print(f"  • Total samples: {total_samples:,}")
    print(f"  • Average reasoning steps: {np.mean(all_steps):.1f}")
    print(f"  • Median reasoning steps: {np.median(all_steps):.1f}")
    print(f"  • Step range: {np.min(all_steps)} - {np.max(all_steps)}")
    print("=" * 100)

if __name__ == "__main__":
    main()
