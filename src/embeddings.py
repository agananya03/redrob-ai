"""
embeddings.py

Generates text embeddings for candidates and job descriptions, with caching.
"""

import os
import pickle
import hashlib
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EmbeddingScorer:
    def __init__(self, cache_dir='data/processed/embeddings'):
        """
        Initializes the EmbeddingScorer with a model and caching directory.
        
        Args:
            cache_dir (str): Path to directory where embedding cache is saved.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = 'data/processed/embeddings/embedding_cache.pkl'
        
        # Load existing cache
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
                if not isinstance(self._cache, dict):
                    self._cache = {}
            except Exception as e:
                logger.warning(f"Could not load existing cache, initializing empty cache: {e}")
                self._cache = {}
        else:
            self._cache = {}
            
        # Initialize sentence-transformers model
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Retrieves the embedding for a given text, computing and caching it if not present.
        
        Args:
            text (str): Text to embed.
            
        Returns:
            np.ndarray: Embedding vector.
        """
        if not isinstance(text, str):
            text = ""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in self._cache:
            return self._cache[text_hash]
            
        # Load model lazily
        if self.model is None:
            self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        emb = self.model.encode(text, normalize_embeddings=True)
        self._cache[text_hash] = emb
        return emb

    def save_cache(self):
        """
        Saves the current embedding cache dictionary to disk.
        """
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")

    def embed_jd(self, jd_text: str) -> np.ndarray:
        """
        Embeds the job description text.
        
        Args:
            jd_text (str): Job description text.
            
        Returns:
            np.ndarray: Embedding vector.
        """
        # BGE models require a specific prefix for queries (but not for documents)
        query_instruction = "Represent this sentence for searching relevant passages: "
        processed_jd_text = query_instruction + jd_text
        
        emb = self._get_embedding(processed_jd_text)
        # Note: We don't cache JD since it changes, but we could in the future
        return emb

    def embed_candidates(self, profiles: list[dict],
                         batch_size: int = 64) -> dict[str, np.ndarray]:
        """
        Embeds a list of candidate profiles.
        """
        if self.model is None:
            self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            
        result = {}
        show_progress = len(profiles) > batch_size
        needs_save = False
        processed_since_save = 0
        save_every = 5000
        
        cache_misses = 0
        
        for i in tqdm(range(0, len(profiles), batch_size), disable=not show_progress, desc="Embedding candidates"):
            batch_profiles = profiles[i:i+batch_size]
            to_encode_texts = []
            to_encode_cids = []
            to_encode_hashes = []
            
            for p in batch_profiles:
                cid = p.get('candidate_id')
                summary = p.get('profile_summary', '')
                if not isinstance(summary, str):
                    summary = ""
                text_hash = hashlib.md5(summary.encode('utf-8')).hexdigest()
                
                if text_hash not in self._cache:
                    cache_misses += 1
                    to_encode_texts.append(summary)
                    to_encode_cids.append(cid)
                    to_encode_hashes.append(text_hash)
                else:
                    result[cid] = self._cache[text_hash]
                    
            if to_encode_texts:
                embs = self.model.encode(to_encode_texts, batch_size=batch_size, normalize_embeddings=True)
                for cid, hsh, emb in zip(to_encode_cids, to_encode_hashes, embs):
                    self._cache[hsh] = emb
                    result[cid] = emb
                needs_save = True
                processed_since_save += len(to_encode_texts)
                
            if needs_save and processed_since_save >= save_every:
                self.save_cache()
                processed_since_save = 0
                needs_save = False

        if needs_save:
            self.save_cache()
            
        print(f"Embedding cache misses: {cache_misses} out of {len(profiles)}")
            
        return result

    def score(self, jd_embedding: np.ndarray,
              candidate_embeddings: dict[str, np.ndarray]
              ) -> dict[str, float]:
        """
        Calculates cosine similarity scores between the JD and all candidate embeddings.
        
        Args:
            jd_embedding (np.ndarray): Embedding of the JD.
            candidate_embeddings (dict[str, np.ndarray]): Mapping of candidate_id to embedding.
            
        Returns:
            dict[str, float]: Mapping of candidate_id to semantic match score (0.0-1.0).
        """
        results = {}
        if jd_embedding is None or not candidate_embeddings:
            return results
            
        norm_jd = np.linalg.norm(jd_embedding)
        if norm_jd == 0:
            return {cid: 0.0 for cid in candidate_embeddings}
            
        for cid, cand_emb in candidate_embeddings.items():
            if cand_emb is None:
                results[cid] = 0.0
                continue
                
            norm_cand = np.linalg.norm(cand_emb)
            if norm_cand == 0:
                results[cid] = 0.0
                continue
                
            dot_product = np.dot(jd_embedding, cand_emb)
            similarity = dot_product / (norm_jd * norm_cand)
            similarity_clipped = float(np.clip(similarity, 0.0, 1.0))
            results[cid] = similarity_clipped
            
        return results

    def save_cache(self):
        """
        Saves the current embedding cache dictionary to disk.
        """
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
