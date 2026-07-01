import time
import json
from src.preprocessing import build_candidate_profile
from src.embeddings import EmbeddingScorer

print("Loading 100K data...")
profiles = []
with open('data/raw/candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        profiles.append(build_candidate_profile(c))

print(f"Encoding {len(profiles)} candidates...")
start = time.time()
scorer = EmbeddingScorer()
scorer.embed_candidates(profiles, batch_size=256)
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")
