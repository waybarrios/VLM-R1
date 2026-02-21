#!/usr/bin/env python3
import sys, json, re

def extract_answer_from_json(response):
    response_cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
    try:
        parsed = json.loads(response_cleaned)
        if isinstance(parsed, dict) and 'answer' in parsed:
            return parsed['answer'].strip()
    except:
        pass

    first_brace = response_cleaned.find('{')
    if first_brace != -1:
        brace_count = 0
        for idx in range(first_brace, len(response_cleaned)):
            if response_cleaned[idx] == '{':
                brace_count += 1
            elif response_cleaned[idx] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = response_cleaned[first_brace:idx+1]
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and 'answer' in parsed:
                            return parsed['answer'].strip()
                    except:
                        pass
                    break

    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response_cleaned)
    if answer_match:
        return answer_match.group(1).strip()

    return response.strip()

samples_file = sys.argv[1]
errors = []
patterns = {}

with open(samples_file, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        sample = json.loads(line)
        raw = sample['filtered_resps'][0]
        answer_text = extract_answer_from_json(raw)
        gt = sample['target']

        # Categorize problematic patterns
        if ':' in answer_text and answer_text[0] in 'ABCD':
            pattern = 'Letter + colon + text'
        elif len(answer_text) > 10:
            pattern = 'Long answer (>10 chars)'
        elif answer_text and answer_text[0] not in 'ABCD':
            pattern = 'Does not start with letter'
        elif ')' in answer_text:
            pattern = 'Contains parenthesis'
        else:
            continue

        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append({
            'id': sample['doc_id'],
            'answer': answer_text,
            'gt': gt
        })

print(f"\nProblematic Answer Patterns:\n{'='*70}")
for pattern, samples in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"\n{pattern}: {len(samples)} samples")
    for s in samples[:5]:
        print(f"  Q{s['id']:4d}: '{s['answer'][:70]}' (gt={s['gt']})")
    if len(samples) > 5:
        print(f"  ... and {len(samples)-5} more")
