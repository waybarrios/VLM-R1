#!/usr/bin/env python3
"""
Extract a specific sample from the dataset for paper figure
Saves image and prints reasoning steps
"""
import pyarrow as pa
import json
from PIL import Image
import io
import sys

def extract_sample(sample_idx: int = 3):
    """Extract sample from dataset"""

    # Load first arrow file (contains samples 0-N)
    arrow_file = '/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27/data-00000-of-00004.arrow'

    # Use open_stream for IPC streaming format
    stream = pa.ipc.open_stream(arrow_file)
    table = stream.read_all()

    # Check if sample is in this file
    if sample_idx >= table.num_rows:
        print(f"Sample {sample_idx} not in first arrow file")
        return None

    # Get the specific row
    sample = table.slice(sample_idx, 1).to_pydict()

    # Extract data
    result = {
        'question': sample['question'][0],
        'answer': sample['answer'][0],
        'reasoning_steps': sample['reference_steps'][0],
    }

    # Extract image if available
    if 'image' in sample and sample['image'][0] is not None:
        image_bytes = sample['image'][0]['bytes']
        result['image'] = Image.open(io.BytesIO(image_bytes))

    return result

if __name__ == '__main__':
    sample_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print(f"Extracting sample {sample_idx}...")
    sample = extract_sample(sample_idx)

    if sample:
        print(f"\nQuestion: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"\nReasoning Steps ({len(sample['reasoning_steps'])} total):")
        for i, step in enumerate(sample['reasoning_steps'], 1):
            print(f"{i}. {step}")

        # Save image if available
        if 'image' in sample:
            output_path = f'/gpudata3/Wayner/paper_reasoning/images/dataset_example_sample{sample_idx}.jpg'
            sample['image'].save(output_path, 'JPEG', quality=95)
            print(f"\n✓ Image saved to: {output_path}")
        else:
            print("\n⚠️  No image found in sample")
    else:
        print(f"✗ Failed to extract sample {sample_idx}")
