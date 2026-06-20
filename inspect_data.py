import json
import csv
import os

path = r"c:\Users\parva\OneDrive\Desktop\india.runs\candidate-ranker\data\raw\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"

def inspect_json(filepath):
    print(f"\n=== {os.path.basename(filepath)} ===")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"Type: List, Length: {len(data)}")
        if data:
            print("Keys:", list(data[0].keys()))
            print("Sample 3 rows:")
            print(json.dumps(data[:3], indent=2)[:2000] + "\n... (truncated)") 
    elif isinstance(data, dict):
        print(f"Type: Dict, Keys: {list(data.keys())}")
        print("Sample 3 values:")
        for k, v in list(data.items())[:3]:
            print(f"{k}: {str(v)[:500]}")

def inspect_jsonl(filepath):
    print(f"\n=== {os.path.basename(filepath)} ===")
    count = 0
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if count < 3:
                samples.append(json.loads(line))
            count += 1
    print(f"Lines count: {count}")
    if samples:
        print("Keys:", list(samples[0].keys()))
        print("Sample 3 rows:")
        print(json.dumps(samples, indent=2)[:2000] + "\n... (truncated)")

def inspect_csv(filepath):
    print(f"\n=== {os.path.basename(filepath)} ===")
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print("Columns:", header)
        rows = []
        count = 1
        for row in reader:
            if len(rows) < 3:
                rows.append(row)
            count += 1
    print(f"Rows count: {count}")
    print("Sample 3 rows:")
    for r in rows:
        print(r)

if __name__ == "__main__":
    inspect_json(os.path.join(path, "sample_candidates.json"))
    inspect_csv(os.path.join(path, "sample_submission.csv"))
    inspect_jsonl(os.path.join(path, "candidates.jsonl"))
    inspect_json(os.path.join(path, "candidate_schema.json"))
