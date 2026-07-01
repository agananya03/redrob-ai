import pickle
with open('data/processed/embeddings/embedding_cache.pkl', 'rb') as f:
    data = pickle.load(f)
print("Cached embeddings:", len(data))
