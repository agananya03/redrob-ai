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
        self.cache_path = os.path.join(self.cache_dir, 'embedding_cache.pkl')
        
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
            
        emb = self.model.encode(text)
        self._cache[text_hash] = emb
        return emb

    def embed_jd(self, jd_text: str) -> np.ndarray:
        """
        Embeds the job description and saves the cache to disk.
        
        Args:
            jd_text (str): Job description text.
            
        Returns:
            np.ndarray: Embedding vector.
        """
        # BGE models require a specific prefix for queries (but not for documents)
        query_instruction = "Represent this sentence for searching relevant passages: "
        processed_jd_text = query_instruction + jd_text
        
        emb = self._get_embedding(processed_jd_text)
        self.save_cache()
        return emb

    def embed_candidates(self, profiles: list[dict],
                         batch_size: int = 64) -> dict[str, np.ndarray]:
        """
        Embeds profile summaries for a list of candidate profiles in batches.
        
        Args:
            profiles (list[dict]): Candidate profiles containing 'candidate_id' and 'profile_summary'.
            batch_size (int): Size of batches to process.
            
        Returns:
            dict[str, np.ndarray]: Dictionary mapping candidate_id to their embedding.
        """
        result = {}
        if not profiles:
            return result
            
        show_progress = len(profiles) > batch_size
        
        for i in tqdm(range(0, len(profiles), batch_size), disable=not show_progress, desc="Embedding candidates"):
            batch_profiles = profiles[i:i+batch_size]
            to_encode_texts = []
            to_encode_hashes = []
            
            for p in batch_profiles:
                summary = p.get('profile_summary', '')
                text_hash = hashlib.md5(summary.encode('utf-8')).hexdigest()
                
                if text_hash not in self._cache:
                    to_encode_texts.append(summary)
                    to_encode_hashes.append(text_hash)
                    
            if to_encode_texts:
                embs = self.model.encode(to_encode_texts, batch_size=batch_size)
                for h, emb in zip(to_encode_hashes, embs):
                    self._cache[h] = emb
                self.save_cache()
                
            for p in batch_profiles:
                cid = p.get('candidate_id', 'UNKNOWN')
                summary = p.get('profile_summary', '')
                text_hash = hashlib.md5(summary.encode('utf-8')).hexdigest()
                result[cid] = self._cache[text_hash]
                
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
