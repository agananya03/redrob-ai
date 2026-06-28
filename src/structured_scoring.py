"""
structured_scoring.py

Calculates candidate ranking scores based on structured data.
"""

import logging
import re
from datetime import datetime, date
from src.preprocessing import extract_platform_signals

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    'skill':            0.30,
    'experience':       0.20,
    'education':        0.10,
    'trajectory':       0.20,
    'platform_signal':  0.20,
}

class StructuredScorer:
    def __init__(self):
        self._jd_cache = {}
        # Precompile regular expressions for efficiency
        self.degree_ms_re = re.compile(r'\bms\b|\bm\.s\b', re.IGNORECASE)
        self.degree_be_re = re.compile(r'\bbe\b|\bbs\b', re.IGNORECASE)
        self.field_cs_re = re.compile(r'\bcs\b|\bai\b|\bml\b', re.IGNORECASE)
        self.it_re = re.compile(r'\bit\b', re.IGNORECASE)

    def _get_jd_info(self, jd_text: str):
        if jd_text in self._jd_cache:
            return self._jd_cache[jd_text]

        known_tech_terms = [
            'python', 'sql', 'spark', 'kafka', 'tensorflow', 'pytorch', 'llm', 'rag',
            'mlops', 'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'fastapi', 'nlp',
            'java', 'c++', 'go', 'rust', 'javascript', 'typescript', 'react', 'html',
            'css', 'hadoop', 'airflow', 'nosql', 'mongodb', 'postgresql', 'mysql',
            'redis', 'pyspark', 'dbt', 'snowflake', 'scala', 'keras', 'scikit-learn',
            'pandas', 'numpy'
        ]
        
        jd_text_lower = jd_text.lower()
        jd_skills = set()
        for term in known_tech_terms:
            escaped = re.escape(term)
            if re.search(r'\b' + escaped + r'\b', jd_text_lower):
                jd_skills.add(term)
            elif term in ['c++', 'c#'] and term in jd_text_lower:
                jd_skills.add(term)

        # Parse experience requirement from jd_text
        match_range = re.search(r'(\d+)\s*-\s*(\d+)\s*years', jd_text, re.IGNORECASE)
        match_plus = re.search(r'(\d+)\+\s*years', jd_text, re.IGNORECASE)
        match_at_least = re.search(r'at least\s+(\d+)\s*years', jd_text, re.IGNORECASE)

        min_required = 5
        if match_range:
            min_required = int(match_range.group(1))
        elif match_plus:
            min_required = int(match_plus.group(1))
        elif match_at_least:
            min_required = int(match_at_least.group(1))

        info = (jd_skills, min_required)
        self._jd_cache[jd_text] = info
        return info

    def score(self, candidate_dict: dict, jd_text: str, weights: dict = None) -> dict:
        """
        Calculates structured sub-scores and the total score for a candidate.
        
        Args:
            candidate_dict (dict): Combined or raw/profile candidate dictionary.
            jd_text (str): Job description text.
            weights (dict, optional): Custom weights mapping. Defaults to None.
            
        Returns:
            dict: Dictionary containing sub-scores and the total score.
        """
        if not isinstance(candidate_dict, dict):
            candidate_dict = {}
        if not isinstance(jd_text, str):
            jd_text = ""

        # Validate weights
        if weights is None:
            weights = DEFAULT_WEIGHTS
        else:
            if not isinstance(weights, dict):
                raise ValueError("weights must be a dictionary")
            required_keys = {'skill', 'experience', 'education', 'trajectory', 'platform_signal'}
            if not required_keys.issubset(weights.keys()):
                raise ValueError(f"weights dictionary must contain all required keys: {required_keys}")
            if abs(sum(weights.values()) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")

        # Get precomputed jd info
        jd_skills, min_required = self._get_jd_info(jd_text)

        # -------------------------------------------------------------
        # SUB-SCORE 1: skill_match_score (0.0-1.0)
        # -------------------------------------------------------------
        candidate_skills = candidate_dict.get('skills')
        if candidate_skills is None:
            # Fallback if profile format skill_names is present
            skill_names = candidate_dict.get('skill_names') or []
            candidate_skills = [{'name': name} for name in skill_names]

        matched_score = 0.0
        for s in candidate_skills:
            if not isinstance(s, dict) or 'name' not in s or s['name'] is None:
                continue
            skill_name = str(s['name']).lower()
            
            # Check for partial case-insensitive match
            matched = False
            for jd_s in jd_skills:
                if jd_s in skill_name or skill_name in jd_s:
                    matched = True
                    break
            
            if matched:
                # Proficiency weight
                prof = str(s.get('proficiency') or '').lower().strip()
                if prof == 'beginner':
                    prof_weight = 0.40
                elif prof == 'intermediate':
                    prof_weight = 0.60
                elif prof == 'advanced':
                    prof_weight = 0.85
                elif prof == 'expert':
                    prof_weight = 1.00
                else:
                    prof_weight = 0.50
                
                # Confidence multiplier
                endorsements = s.get('endorsements')
                if endorsements is None:
                    endorsements = 0
                try:
                    endorsements = float(endorsements)
                except (ValueError, TypeError):
                    endorsements = 0.0

                duration_months = s.get('duration_months')
                if duration_months is None:
                    duration_months = 0
                try:
                    duration_months = float(duration_months)
                except (ValueError, TypeError):
                    duration_months = 0.0

                confidence_mult = min(1.0, endorsements * 0.05 + duration_months / 60)
                matched_score += (prof_weight * confidence_mult)

        skill_match_score = min(1.0, matched_score / max(len(jd_skills), 1))

        # -------------------------------------------------------------
        # SUB-SCORE 2: experience_score (0.0-1.0)
        # -------------------------------------------------------------
        # Retrieve years of experience
        years = candidate_dict.get('years_of_experience')
        if years is None:
            years = candidate_dict.get('profile', {}).get('years_of_experience')
        
        if years is None:
            years_val = 0.0
        else:
            try:
                years_val = float(years)
            except (ValueError, TypeError):
                years_val = 0.0

        if years_val > min_required + 5:
            experience_score = 0.85
        elif years_val >= min_required:
            experience_score = 1.00
        elif years_val >= min_required - 2:
            experience_score = 0.70
        elif years_val >= min_required - 4:
            experience_score = 0.40
        else:
            experience_score = 0.15

        # -------------------------------------------------------------
        # SUB-SCORE 3: education_score (0.0-1.0)
        # -------------------------------------------------------------
        def get_degree_score(degree_str: str) -> float:
            if not degree_str:
                return 0.30
            d = degree_str.lower()
            if 'phd' in d or 'ph.d' in d or 'doctor' in d:
                return 1.00
            if 'master' in d or 'm.tech' in d or 'mtech' in d or 'mba' in d:
                return 0.85
            if self.degree_ms_re.search(d):
                return 0.85
            if 'bachelor' in d or 'b.tech' in d or 'btech' in d or 'b.e' in d or 'b.s' in d:
                return 0.70
            if self.degree_be_re.search(d):
                return 0.70
            if 'diploma' in d or 'associate' in d:
                return 0.45
            return 0.30

        def get_field_multiplier(field_str: str) -> float:
            if not field_str:
                return 0.45
            f = field_str.lower()
            if 'computer' in f or 'software' in f or 'data science' in f or 'artificial intelligence' in f:
                return 1.00
            if self.field_cs_re.search(f):
                return 1.00
            if 'math' in f or 'statistic' in f or 'physic' in f:
                return 0.85
            if 'electronic' in f or 'electric' in f:
                return 0.70
            stem_keywords = ['engineering', 'science', 'technology', 'tech', 'mechanical', 'civil', 'chemical', 'biotech', 'information']
            for kw in stem_keywords:
                if kw in f:
                    return 0.60
            return 0.45

        def get_tier_multiplier(tier_str: str) -> float:
            if not tier_str:
                return 0.90
            t = tier_str.lower().strip()
            if 'tier_1' in t or 'tier1' in t:
                return 1.00
            if 'tier_2' in t or 'tier2' in t:
                return 0.93
            if 'tier_3' in t or 'tier3' in t:
                return 0.86
            if 'tier_4' in t or 'tier4' in t:
                return 0.80
            return 0.90

        education_list = candidate_dict.get('education', [])
        if not education_list:
            education_score = 0.40
        else:
            entry_scores = []
            for edu in education_list:
                if not isinstance(edu, dict):
                    continue
                deg = edu.get('degree')
                field = edu.get('field_of_study')
                tier = edu.get('tier')
                
                deg_score = get_degree_score(deg)
                field_mult = get_field_multiplier(field)
                tier_mult = get_tier_multiplier(tier)
                
                entry_scores.append(deg_score * field_mult * tier_mult)
            education_score = max(entry_scores) if entry_scores else 0.40

        # -------------------------------------------------------------
        # SUB-SCORE 4: trajectory_score (0.0-1.0)
        # -------------------------------------------------------------
        career_history = candidate_dict.get('career_history', [])
        valid_roles = []
        for role in career_history:
            if isinstance(role, dict) and role.get('title'):
                valid_roles.append(role)

        def get_start_date_key_asc(r):
            start_date_str = r.get('start_date')
            if not start_date_str:
                return (1, datetime.min)
            try:
                dt = datetime.strptime(str(start_date_str), "%Y-%m-%d")
                return (0, dt)
            except Exception:
                return (1, datetime.min)

        valid_roles_sorted = sorted(valid_roles, key=get_start_date_key_asc)

        if not valid_roles_sorted:
            progression = 0.5
            stability = 0.5
            relevance = 0.5
        else:
            # A. Progression
            def get_seniority_level(title: str) -> int:
                if not title:
                    return 3
                t = title.lower()
                levels = []
                if 'cto' in t or 'cpo' in t or 'founder' in t:
                    levels.append(8)
                if 'director' in t or 'vp' in t:
                    levels.append(7)
                if 'principal' in t or 'manager' in t:
                    levels.append(6)
                if 'lead' in t or 'staff' in t:
                    levels.append(5)
                if 'senior' in t:
                    levels.append(4)
                if 'mid-level' in t or 'engineer' in t:
                    levels.append(3)
                if 'junior' in t or 'associate' in t:
                    levels.append(2)
                if 'intern' in t or 'trainee' in t:
                    levels.append(1)
                return max(levels) if levels else 3

            num_roles = len(valid_roles_sorted)
            if num_roles == 1:
                progression = 0.7
            else:
                levels = [get_seniority_level(r.get('title')) for r in valid_roles_sorted]
                initial_lvl = levels[0]
                final_lvl = levels[-1]
                if num_roles >= 3 and len(set(levels)) == 1:
                    progression = 0.6
                elif final_lvl < initial_lvl:
                    progression = 0.4
                else:
                    progression = 1.0

            # B. Stability
            durations = []
            for r in valid_roles_sorted:
                dur = r.get('duration_months')
                if dur is None:
                    durations.append(0.0)
                else:
                    try:
                        durations.append(float(dur))
                    except (ValueError, TypeError):
                        durations.append(0.0)
            avg_dur = sum(durations) / len(durations) if durations else 0.0

            if avg_dur >= 24:
                stability = 1.0
            elif avg_dur >= 18:
                stability = 0.85
            elif avg_dur >= 12:
                stability = 0.65
            elif avg_dur >= 6:
                stability = 0.40
            else:
                stability = 0.20

            # C. Relevance
            relevant_keywords = {
                'technology', 'software', 'it', 'artificial intelligence',
                'data', 'saas', 'internet', 'fintech', 'edtech'
            }
            relevant_count = 0
            for r in valid_roles_sorted:
                ind = r.get('industry')
                if ind:
                    ind_lower = str(ind).lower()
                    matched_ind = False
                    for kw in relevant_keywords:
                        if kw == 'it':
                            if self.it_re.search(ind_lower):
                                matched_ind = True
                                break
                        else:
                            if kw in ind_lower:
                                matched_ind = True
                                break
                    if matched_ind:
                        relevant_count += 1
            relevance = relevant_count / num_roles

        trajectory_score = (progression + stability + relevance) / 3

        # -------------------------------------------------------------
        # SUB-SCORE 5: platform_signal_score (0.0-1.0)
        # -------------------------------------------------------------
        # Pass candidate through extract_platform_signals to flat-extract and clean sentinels
        signals = extract_platform_signals(candidate_dict)

        # 1. open_to_work_flag
        otw = signals.get('open_to_work_flag')
        if otw is None:
            otw_score = 0.5
        elif otw is True:
            otw_score = 1.0
        else:
            otw_score = 0.3

        # 2. last_active_date recency
        active_str = signals.get('last_active_date')
        if not active_str:
            recency_score = 0.5
        else:
            try:
                active_date = datetime.strptime(str(active_str).strip(), "%Y-%m-%d").date()
                today = date.today()
                days = (today - active_date).days
                if days < 0:
                    days = 0
                if days <= 7:
                    recency_score = 1.0
                elif days <= 30:
                    recency_score = 0.85
                elif days <= 90:
                    recency_score = 0.60
                elif days <= 180:
                    recency_score = 0.40
                else:
                    recency_score = 0.20
            except Exception:
                recency_score = 0.5

        # 3. profile_completeness_score
        completeness = signals.get('profile_completeness_score')
        if completeness is None:
            completeness_score = 0.5
        else:
            try:
                val = float(completeness)
                if val > 1.0:
                    val /= 100.0
                completeness_score = max(0.0, min(1.0, val))
            except Exception:
                completeness_score = 0.5

        # 4. recruiter_response_rate
        resp_rate = signals.get('recruiter_response_rate')
        if resp_rate is None:
            resp_rate_score = 0.5
        else:
            try:
                val = float(resp_rate)
                if val > 1.0:
                    val /= 100.0
                resp_rate_score = max(0.0, min(1.0, val))
            except Exception:
                resp_rate_score = 0.5

        # 5. interview_completion_rate
        int_rate = signals.get('interview_completion_rate')
        if int_rate is None:
            int_rate_score = 0.5
        else:
            try:
                val = float(int_rate)
                if val > 1.0:
                    val /= 100.0
                int_rate_score = max(0.0, min(1.0, val))
            except Exception:
                int_rate_score = 0.5

        # 6. offer_acceptance_rate
        off_rate = signals.get('offer_acceptance_rate')
        if off_rate is None:
            off_rate_score = 0.5
        else:
            try:
                val = float(off_rate)
                if val > 1.0:
                    val /= 100.0
                off_rate_score = max(0.0, min(1.0, val))
            except Exception:
                off_rate_score = 0.5

        platform_signal_score = (
            0.25 * otw_score +
            0.20 * recency_score +
            0.15 * completeness_score +
            0.20 * resp_rate_score +
            0.10 * int_rate_score +
            0.10 * off_rate_score
        )

        # -------------------------------------------------------------
        # WEIGHTED TOTAL
        # -------------------------------------------------------------
        total_score = (
            weights['skill'] * skill_match_score +
            weights['experience'] * experience_score +
            weights['education'] * education_score +
            weights['trajectory'] * trajectory_score +
            weights['platform_signal'] * platform_signal_score
        )

        return {
            'skill_match_score': skill_match_score,
            'experience_score': experience_score,
            'education_score': education_score,
            'trajectory_score': trajectory_score,
            'platform_signal_score': platform_signal_score,
            'total_score': total_score
        }
