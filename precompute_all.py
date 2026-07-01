import time
import json
from src.embeddings import EmbeddingScorer

print("Loading 100K data...")
profiles = []
with open('data/raw/candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        profiles.append({'candidate_id': c.get('candidate_id'), 'profile_summary': c.get('profile', {}).get('profile_summary', '')})

print(f"Encoding {len(profiles)} candidates...")
start = time.time()
scorer = EmbeddingScorer()
scorer.embed_candidates(profiles, batch_size=128)
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")
