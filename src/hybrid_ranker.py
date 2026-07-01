"""
hybrid_ranker.py

Combines structured scores and embedding-based semantic scores to rank candidates.
"""

import os
import time
import logging
import pandas as pd

logger = logging.getLogger(__name__)
from src.data_loader import load_single_candidate
from src.preprocessing import build_candidate_profile
from src.structured_scoring import StructuredScorer
from src.embeddings import EmbeddingScorer
from src.ltr_ranker import load_training_data, train_ranker, predict_scores, save_model, load_model

class HybridRanker:
    def __init__(self,
                 structured_weight: float = 0.60,
                 semantic_weight: float = 0.40,
                 structured_score_weights: dict = None):
        """
        Initializes the HybridRanker with scorer components and overall weights.
        
        Args:
            structured_weight (float): Weight for structured scores (0.0-1.0).
            semantic_weight (float): Weight for semantic match scores (0.0-1.0).
            structured_score_weights (dict, optional): Weights within StructuredScorer. Defaults to None.
        """
        if abs(structured_weight + semantic_weight - 1.0) > 1e-6:
            raise ValueError("structured_weight and semantic_weight must sum to 1.0")
            
        self.structured_weight = structured_weight
        self.semantic_weight = semantic_weight
        self.structured_score_weights = structured_score_weights
        
        self.structured_scorer = StructuredScorer()
        self.embedding_scorer = EmbeddingScorer()

    def rank(self,
             candidates: list[dict],
             jd_text: str,
             top_n: int = None,
             use_ltr: bool = True,
             use_cross_encoder: bool = False,
             ablation_feature: str = None) -> pd.DataFrame:
        """
        Ranks candidates based on a weighted combination of structured and semantic scores.
        
        Args:
            candidates (list[dict]): Raw candidate list.
            jd_text (str): Job description text.
            top_n (int, optional): Number of top results to return.
            
        Returns:
            pd.DataFrame: DataFrame containing candidates, scores, and ranks.
            
        Raises:
            ValueError: If the candidates list is empty.
        """
        if not candidates:
            raise ValueError("Candidates list is empty.")
            
        # 1. Load single candidates and build profiles
        loaded_candidates = [load_single_candidate(c) for c in candidates]
        profiles = [build_candidate_profile(c) for c in loaded_candidates]
        
        # 2 & 3. Embed JD and candidates
        jd_emb = self.embedding_scorer.embed_jd(jd_text)
        cand_embs = self.embedding_scorer.embed_candidates(profiles)
        
        # 4. Get semantic scores
        semantic_scores = self.embedding_scorer.score(jd_emb, cand_embs)
        
        # 5 & 6. Calculate structured scores and combine with semantic scores
        rows = []
        for raw, prof in zip(loaded_candidates, profiles):
            cid = prof['candidate_id']
            sem_score = semantic_scores.get(cid, 0.0)
            
            if ablation_feature == 'semantic_score':
                sem_score = 0.0
            
            # Combine raw candidate fields and processed profile fields for StructuredScorer
            combined = {**raw, **prof}
            
            struct_res = self.structured_scorer.score(combined, jd_text, weights=self.structured_score_weights)
            
            if ablation_feature in ['platform_signal_score', 'trajectory_score', 'skill_match_score', 'experience_score', 'education_score']:
                struct_res[ablation_feature] = 0.0
                
                # Re-compute structured_total
                weights_to_use = self.structured_score_weights or {
                    'skill': 0.30, 'experience': 0.20, 'education': 0.10,
                    'trajectory': 0.20, 'platform_signal': 0.20
                }
                
                recomputed_total = (
                    weights_to_use['skill'] * struct_res['skill_match_score'] +
                    weights_to_use['experience'] * struct_res['experience_score'] +
                    weights_to_use['education'] * struct_res['education_score'] +
                    weights_to_use['trajectory'] * struct_res['trajectory_score'] +
                    weights_to_use['platform_signal'] * struct_res['platform_signal_score']
                )
                struct_res['total_score'] = recomputed_total
                
            struct_total = struct_res['total_score']
            
            final_score = self.structured_weight * struct_total + self.semantic_weight * sem_score
            
            rows.append({
                'candidate_id': cid,
                'final_score': final_score,
                'skill_match_score': struct_res['skill_match_score'],
                'experience_score': struct_res['experience_score'],
                'education_score': struct_res['education_score'],
                'trajectory_score': struct_res['trajectory_score'],
                'platform_signal_score': struct_res['platform_signal_score'],
                'structured_total': struct_total,
                'semantic_score': sem_score,
                'current_title': prof['current_title'],
                'years_of_experience': prof['years_of_experience'],
                'location': prof['location'],
                'profile_summary': prof.get('profile_summary', '')
            })
            
        df = pd.DataFrame(rows)
        
        labels_path = 'data/processed/relevance_labels.csv'
        model_path = 'data/processed/ltr_model.pkl'
        
        if use_ltr:
            if os.path.exists(labels_path):
                try:
                    model = load_model(model_path)
                except FileNotFoundError:
                    X, y, feature_names = load_training_data(labels_path, df)
                    model = train_ranker(X, y, feature_names)
                    save_model(model, model_path)
                
                ltr_scores = predict_scores(model, df)
                df['final_score'] = df['candidate_id'].map(ltr_scores)
                logger.info('Using LightGBM LTR for final ranking')
            else:
                logger.warning('relevance_labels.csv not found, falling back to weighted sum. Run label_candidates.py first.')
                
        if use_cross_encoder:
            start_ce = time.time()
            from sentence_transformers import CrossEncoder
            logger.info("Running CrossEncoder on top 300 candidates...")
            
            # Sort first by final_score descending to get the best candidates
            df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
            
            top_k = min(300, len(df))
            top_df = df.head(top_k).copy()
            rest_df = df.iloc[top_k:].copy()
            
            ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [[jd_text, str(summary)] for summary in top_df['profile_summary']]
            
            ce_scores = ce_model.predict(pairs)
            
            # Min-Max normalize ce_scores to [10.0, 20.0] to stay strictly above LightGBM scores
            import numpy as np
            ce_min, ce_max = np.min(ce_scores), np.max(ce_scores)
            if ce_max > ce_min:
                norm_ce = 10.0 + 10.0 * (ce_scores - ce_min) / (ce_max - ce_min)
            else:
                norm_ce = 10.0
                
            top_df['final_score'] = norm_ce
            
            # Combine back
            df = pd.concat([top_df, rest_df]).reset_index(drop=True)
            
            ce_time = time.time() - start_ce
            logger.info(f"Cross-encoder reranking took {ce_time:.2f} seconds.")

        df['calibrated_score'] = df['final_score'].rank(pct=True).round(4)
        
        # 8 & 9. Sort descending and add rank
        df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        
        # Reorder columns to put 'rank' first
        cols_order = [
            'rank', 'candidate_id', 'final_score', 'calibrated_score',
            'skill_match_score', 'experience_score', 'education_score',
            'trajectory_score', 'platform_signal_score',
            'structured_total', 'semantic_score',
            'current_title', 'years_of_experience', 'location', 'profile_summary'
        ]
        df = df[cols_order]
        
        # 10. Limit to top_n
        if top_n is not None:
            df = df.head(top_n)
            
        return df

    def llm_rerank(self,
                   top_candidates: pd.DataFrame,
                   jd_text: str,
                   n: int = 15,
                   skip_llm: bool = False) -> pd.DataFrame:
        """
        Appends LLM-generated reasoning for candidate ranking.
        
        Args:
            top_candidates (pd.DataFrame): DataFrame of ranked candidates.
            jd_text (str): Job description text.
            n (int, optional): Number of candidates to rerank.
            skip_llm (bool, optional): If True, skips LLM calls and fills with empty strings.
            
        Returns:
            pd.DataFrame: DataFrame with the added 'reasoning' column.
        """
        df = top_candidates.copy()
        if skip_llm:
            df['reasoning'] = ''
            return df
            
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Skipping LLM rerank.")
            df['reasoning'] = 'N/A'
            return df
            
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
        except ImportError:
            logger.warning("groq SDK not installed. Skipping LLM rerank.")
            df['reasoning'] = 'N/A'
            return df

        system_prompt = (
            "You are an expert technical recruiter evaluating candidates for "
            "a Senior AI Engineer role. Be specific — reference actual skills, "
            "years, company names, and titles from the candidate's profile. "
            "Never give generic praise."
        )

        reasonings = []
        total = len(df)
        for idx, row in df.iterrows():
            logger.info(f"Generating reasoning for candidate {len(reasonings)+1}/{total}...")
            
            candidate_id = row.get('candidate_id', 'Unknown')
            current_title = row.get('current_title', 'Unknown')
            years_of_experience = row.get('years_of_experience', 0)
            skill_match_score = row.get('skill_match_score', 0.0)
            experience_score = row.get('experience_score', 0.0)
            trajectory_score = row.get('trajectory_score', 0.0)
            platform_signal_score = row.get('platform_signal_score', 0.0)
            profile_summary = str(row.get('profile_summary', ''))
            
            user_prompt = f"Job Description:\n{jd_text[:2000]}\n\nCandidate profile:\n- ID: {candidate_id}\n- Title: {current_title}\n- Experience: {years_of_experience} years\n- Score breakdown: skill={skill_match_score:.2f}, exp={experience_score:.2f}, trajectory={trajectory_score:.2f}, platform={platform_signal_score:.2f}\n- Profile summary (first 800 chars): {profile_summary[:800]}\n\nWrite ONE sentence (max 25 words) explaining why this candidate ranks where they do. Start with their strongest relevant signal."

            try:
                response = client.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    max_tokens=80,
                    temperature=0.3,
                )
                reasoning = response.choices[0].message.content
                reasoning = reasoning.strip()[:200]
            except Exception as e:
                logger.error(f"API error for {candidate_id}: {e}")
                reasoning = "API error"
                
            reasonings.append(reasoning)
            time.sleep(0.5)
            
        df['reasoning'] = reasonings
        return df
