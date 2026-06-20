"""
preprocessing.py

Handles candidate profile building and platform signal extraction.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def build_candidate_profile(candidate: dict) -> dict:
    """
    Builds a clean text representation of the candidate for embedding.

    Steps:
    1. Pull from profile{}:
         headline, summary, current_title, current_company,
         current_industry, years_of_experience, location, country

    2. Build career_text from career_history[] sorted by start_date
       DESCENDING (most recent first). For each role:
         '{title} at {company} ({duration_months} mo): {description}'
       - Safely skip description if None or empty
       - Safely skip the whole role if title or company is missing
       - Parse start_date as YYYY-MM-DD; if parsing fails, keep the
         role but sort it last

    3. Build skills_text:
         'Skills: ' + comma-joined list of skill['name'] values
       Skip any skill entry missing a 'name' key.

    4. Concatenate into profile_summary (newline-separated):
         headline
         summary
         '{current_title} at {current_company} ({current_industry})'
         career_text
         skills_text

    Args:
        candidate (dict): The raw candidate dictionary.
        
    Returns:
        dict: A dictionary containing extracted profile details.
    """
    try:
        if not isinstance(candidate, dict):
            candidate = {}
            
        candidate_id = candidate.get('candidate_id')
        if candidate_id is None:
            candidate_id = 'UNKNOWN'
        else:
            candidate_id = str(candidate_id)
            
        profile = candidate.get('profile') or {}
        
        # 1. Pull from profile
        headline = profile.get('headline') or ''
        summary = profile.get('summary') or ''
        current_title = profile.get('current_title')
        current_company = profile.get('current_company')
        current_industry = profile.get('current_industry')
        
        years_of_exp = profile.get('years_of_experience')
        if years_of_exp is not None:
            try:
                years_of_exp = float(years_of_exp)
            except (ValueError, TypeError):
                years_of_exp = None
                
        location = profile.get('location')
        country = profile.get('country')
        
        # 2. Build career_text
        career_history = candidate.get('career_history') or []
        valid_roles = []
        for role in career_history:
            if not isinstance(role, dict):
                continue
            title = role.get('title')
            company = role.get('company')
            if not title or not str(title).strip() or not company or not str(company).strip():
                continue
            
            valid_roles.append(role)
            
        # Sort valid roles by start_date descending (most recent first)
        def get_start_date_key(r):
            start_date_str = r.get('start_date')
            if not start_date_str:
                return (0, datetime.min)
            try:
                dt = datetime.strptime(str(start_date_str), "%Y-%m-%d")
                return (1, dt)
            except Exception:
                return (0, datetime.min)
                
        valid_roles_sorted = sorted(valid_roles, key=get_start_date_key, reverse=True)
        
        career_lines = []
        for role in valid_roles_sorted:
            title = role.get('title')
            company = role.get('company')
            duration = role.get('duration_months')
            if duration is None:
                duration = 0
            
            description = role.get('description')
            if description is not None and str(description).strip():
                desc_str = f": {description}"
            else:
                desc_str = ""
                
            role_line = f"{title} at {company} ({duration} mo){desc_str}"
            career_lines.append(role_line)
            
        career_text = "\n".join(career_lines)
        
        # 3. Build skills_text
        skills = candidate.get('skills') or []
        skill_names = []
        for s in skills:
            if isinstance(s, dict) and 'name' in s:
                name_val = s['name']
                if name_val is not None:
                    skill_names.append(str(name_val))
                    
        skills_text = ""
        if skill_names:
            skills_text = "Skills: " + ", ".join(skill_names)
            
        # 4. Concatenate into profile_summary
        summary_lines = []
        if headline:
            summary_lines.append(headline)
        if summary:
            summary_lines.append(summary)
            
        current_title_str = current_title or ''
        current_company_str = current_company or ''
        current_industry_str = current_industry or ''
        if current_title_str or current_company_str or current_industry_str:
            summary_lines.append(f"{current_title_str} at {current_company_str} ({current_industry_str})")
            
        if career_text:
            summary_lines.append(career_text)
        if skills_text:
            summary_lines.append(skills_text)
            
        profile_summary = "\n".join(summary_lines)
        
        return {
            'candidate_id': candidate_id,
            'profile_summary': profile_summary,
            'years_of_experience': years_of_exp,
            'location': location,
            'country': country,
            'current_title': current_title,
            'current_industry': current_industry,
            'skill_names': skill_names,
        }
    except Exception as e:
        logger.error(f"Error building profile for candidate {candidate.get('candidate_id', 'UNKNOWN') if isinstance(candidate, dict) else 'UNKNOWN'}: {e}")
        return {
            'candidate_id': 'UNKNOWN',
            'profile_summary': '',
            'years_of_experience': None,
            'location': None,
            'country': None,
            'current_title': None,
            'current_industry': None,
            'skill_names': [],
        }

def extract_platform_signals(candidate: dict) -> dict:
    """
    Pulls every field from redrob_signals{} into a flat dict.

    Sentinel rule: ANY numeric field that equals -1 -> replace with None.
    These are NOT low scores; they mean 'data not available'.
    Callers must treat None as neutral (0.5), never as 0.

    Fields to extract from redrob_signals:
      profile_completeness_score, signup_date, last_active_date,
      open_to_work_flag, profile_views_received_30d,
      applications_submitted_30d, recruiter_response_rate,
      avg_response_time_hours, skill_assessment_scores,
      connection_count, endorsements_received, notice_period_days,
      expected_salary_range_inr_lpa (keep as nested dict {min, max}),
      preferred_work_mode, willing_to_relocate, github_activity_score,
      search_appearance_30d, saved_by_recruiters_30d,
      interview_completion_rate, offer_acceptance_rate,
      verified_email, verified_phone, linkedin_connected

    Args:
        candidate (dict): The raw candidate dictionary.
        
    Returns:
        dict: A dictionary containing extracted platform signals.
    """
    fields = [
        'profile_completeness_score', 'signup_date', 'last_active_date',
        'open_to_work_flag', 'profile_views_received_30d',
        'applications_submitted_30d', 'recruiter_response_rate',
        'avg_response_time_hours', 'skill_assessment_scores',
        'connection_count', 'endorsements_received', 'notice_period_days',
        'expected_salary_range_inr_lpa',
        'preferred_work_mode', 'willing_to_relocate', 'github_activity_score',
        'search_appearance_30d', 'saved_by_recruiters_30d',
        'interview_completion_rate', 'offer_acceptance_rate',
        'verified_email', 'verified_phone', 'linkedin_connected'
    ]
    try:
        if not isinstance(candidate, dict):
            candidate = {}
            
        signals = candidate.get('redrob_signals')
        if not isinstance(signals, dict):
            signals = {}
            
        def clean_sentinels(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                if val == -1 or val == -1.0:
                    return None
                return val
            if isinstance(val, dict):
                return {k: clean_sentinels(v) for k, v in val.items()}
            if isinstance(val, list):
                return [clean_sentinels(item) for item in val]
            return val
            
        result = {}
        for field in fields:
            if field not in signals or signals[field] is None:
                result[field] = None
            elif field == 'expected_salary_range_inr_lpa':
                val = signals[field]
                if isinstance(val, dict):
                    min_val = clean_sentinels(val.get('min'))
                    max_val = clean_sentinels(val.get('max'))
                    result[field] = {'min': min_val, 'max': max_val}
                else:
                    result[field] = None
            else:
                result[field] = clean_sentinels(signals[field])
                
        return result
    except Exception as e:
        logger.error(f"Error extracting platform signals for candidate {candidate.get('candidate_id', 'UNKNOWN') if isinstance(candidate, dict) else 'UNKNOWN'}: {e}")
        return {f: None for f in fields}
