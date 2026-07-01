git reset HEAD~1
git lfs track "*.pkl"
git add .gitattributes
git add -f data/processed/ltr_model.pkl
git add -f data/processed/embeddings/embedding_cache.pkl
git add -f data/processed/jd_esco_skills.json
git add -f data/processed/relevance_labels.csv
git commit -m "Include processed models and caches with LFS for HF deployment"
