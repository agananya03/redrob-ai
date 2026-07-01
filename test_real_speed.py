import time
import json
from src.preprocessing import build_candidate_profile
from src.embeddings import EmbeddingScorer

print("Loading data...")
profiles = []
with open('data/raw/candidates.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 1000: break
        c = json.loads(line)
        profiles.append(build_candidate_profile(c))

print("Encoding 1000 real candidates...")
start = time.time()
scorer = EmbeddingScorer()
scorer.embed_candidates(profiles, batch_size=128)
end = time.time()
print(f"Time taken for 1000: {end - start:.2f} seconds")
