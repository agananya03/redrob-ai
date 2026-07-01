git reset HEAD~1
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
git add -f data/models/finetuned_embedder/model.safetensors
git commit -m "Track model with Git LFS"
