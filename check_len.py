import json
with open('data/raw/sample_candidates.json', 'r') as f:
    data = json.load(f)
print(len(data))
