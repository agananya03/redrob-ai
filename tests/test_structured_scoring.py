import pytest
from datetime import datetime, timedelta
from src.structured_scoring import StructuredScorer

def test_zero_skill_match():
    candidate = {
        'candidate_id': 'CAND_01',
        'skills': [
            {'name': 'Excel', 'proficiency': 'expert', 'endorsements': 10, 'duration_months': 36}
        ]
    }
    jd_text = "We need skills in Python, LLM, Spark, and Docker."
    scorer = StructuredScorer()
    scores = scorer.score(candidate, jd_text)
    assert scores['skill_match_score'] < 0.15

def test_perfect_skill_match():
    candidate = {
        'candidate_id': 'CAND_02',
        'skills': [
            {'name': 'Python', 'proficiency': 'expert', 'endorsements': 10, 'duration_months': 60},
            {'name': 'LLM', 'proficiency': 'expert', 'endorsements': 10, 'duration_months': 60},
            {'name': 'Spark', 'proficiency': 'expert', 'endorsements': 10, 'duration_months': 60},
            {'name': 'Docker', 'proficiency': 'expert', 'endorsements': 10, 'duration_months': 60}
        ]
    }
    jd_text = "Python, LLM, Spark, Docker"
    scorer = StructuredScorer()
    scores = scorer.score(candidate, jd_text)
    assert scores['skill_match_score'] > 0.80

def test_under_qualified_experience():
    candidate = {
        'candidate_id': 'CAND_03',
        'years_of_experience': 2.0
    }
    jd_text = "We require at least 8+ years of experience."
    scorer = StructuredScorer()
    scores = scorer.score(candidate, jd_text)
    assert scores['experience_score'] < 0.30

def test_low_platform_signal():
    last_active = (datetime.now() - timedelta(days=540)).strftime("%Y-%m-%d")
    candidate = {
        'candidate_id': 'CAND_04',
        'redrob_signals': {
            'open_to_work_flag': False,
            'last_active_date': last_active,
            'recruiter_response_rate': 0.1
        }
    }
    scorer = StructuredScorer()
    scores = scorer.score(candidate, "Dummy JD")
    assert scores['platform_signal_score'] < 0.40

def test_neutral_sentinels():
    candidate = {
        'candidate_id': 'CAND_05',
        'redrob_signals': {}
    }
    scorer = StructuredScorer()
    scores = scorer.score(candidate, "Dummy JD")
    assert abs(scores['platform_signal_score'] - 0.5) < 0.05
