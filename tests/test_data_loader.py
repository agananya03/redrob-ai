import pytest
from src.data_loader import load_candidates, load_single_candidate, load_job_description
from src.preprocessing import build_candidate_profile, extract_platform_signals

def test_load_sample_candidates():
    records = load_candidates("data/raw/sample_candidates.json")
    assert len(records) == 50
    for r in records:
        assert 'candidate_id' in r

def test_build_profile_no_crash():
    records = load_candidates("data/raw/sample_candidates.json")
    for r in records:
        profile = build_candidate_profile(r)
        assert isinstance(profile['profile_summary'], str)
        assert len(profile['profile_summary']) > 0
        assert isinstance(profile['skill_names'], list)

def test_sentinel_handling():
    candidate = {
        'candidate_id': 'TEST_01',
        'redrob_signals': {
            'github_activity_score': -1,
            'offer_acceptance_rate': -1
        }
    }
    signals = extract_platform_signals(candidate)
    assert signals['github_activity_score'] is None
    assert signals['offer_acceptance_rate'] is None

def test_null_end_date_no_crash():
    candidate = {
        'candidate_id': 'TEST_02',
        'profile': {
            'headline': 'AI Dev',
            'summary': 'Summary text',
            'current_title': 'Engineer',
            'current_company': 'Tech Corp',
            'current_industry': 'Tech'
        },
        'career_history': [
            {
                'company': 'Tech Corp',
                'title': 'Engineer',
                'start_date': '2024-01-01',
                'end_date': None,
                'is_current': True,
                'duration_months': 12,
                'description': 'Working hard'
            }
        ]
    }
    # This should run without any exception
    profile = build_candidate_profile(candidate)
    assert profile['candidate_id'] == 'TEST_02'
    assert 'Engineer at Tech Corp' in profile['profile_summary']

def test_load_job_description():
    jd_text = load_job_description("data/raw/job_description.docx")
    assert isinstance(jd_text, str)
    assert len(jd_text) > 0
    assert 'Senior AI Engineer' in jd_text

def test_missing_fields_safe_defaults():
    empty_cand = {}
    loaded = load_single_candidate(empty_cand)
    assert loaded['candidate_id'] == 'UNKNOWN'
    assert loaded['career_history'] == []
    assert loaded['education'] == []
    assert loaded['skills'] == []
    assert loaded['certifications'] == []
    assert loaded['languages'] == []
    assert loaded['profile'] == {}
    assert loaded['redrob_signals'] == {}
