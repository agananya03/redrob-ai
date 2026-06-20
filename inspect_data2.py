import json
import os
try:
    import docx
except ImportError:
    docx = None

path = r"c:\Users\parva\OneDrive\Desktop\india.runs\candidate-ranker\data\raw\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"

def inspect_docx(filename):
    print(f"\n=== {filename} ===")
    if not docx:
        print("python-docx not installed, skipping.")
        return
    try:
        doc = docx.Document(os.path.join(path, filename))
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        print("Total paragraphs:", len(full_text))
        print("First 3 paragraphs:")
        for p in full_text[:3]:
            print(" -", p)
    except Exception as e:
        print("Error reading docx:", e)

inspect_docx("job_description.docx")

print("\n=== Messiness Check in sample_candidates.json ===")
with open(os.path.join(path, "sample_candidates.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

null_counts = {}
for item in data:
    for k, v in item.items():
        if v is None:
            null_counts[k] = null_counts.get(k, 0) + 1
            
    profile = item.get("profile", {})
    for pk, pv in profile.items():
        if pv is None or str(pv).strip() == "":
            null_counts[f"profile.{pk}"] = null_counts.get(f"profile.{pk}", 0) + 1
            
    for ch in item.get("career_history", []):
        for chk, chv in ch.items():
            if chv is None or str(chv).strip() == "":
                null_counts[f"career_history.{chk}"] = null_counts.get(f"career_history.{chk}", 0) + 1

print("Missing or Null value counts across 50 samples:")
if not null_counts:
    print("  No missing values found!")
for k, v in null_counts.items():
    print(f"  {k}: {v} missing/nulls")
