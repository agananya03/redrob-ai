import json
count = 0
empty = 0
with open('data/raw/candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        summary = c.get('profile', {}).get('profile_summary', '')
        if summary.strip():
            count += 1
        else:
            empty += 1
print(f"Non-empty: {count}, Empty: {empty}")
