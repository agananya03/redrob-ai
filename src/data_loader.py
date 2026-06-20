"""
data_loader.py

Responsible for loading the raw dataset and job description.
"""

import json
import logging
import docx

logger = logging.getLogger(__name__)

def load_candidates(path: str) -> list[dict]:
    """
    Reads candidate data from either:
      .jsonl file — one JSON object per line (full 100k dataset)
      .json file  — JSON array (sample_candidates.json)
    Detects format from file extension.
    Keeps ALL nested fields intact — do NOT flatten.
    Skips and logs (do not crash) any line that fails to parse.
    Returns list of raw candidate dicts.
    """
    path_lower = path.lower()
    records = []
    
    if path_lower.endswith('.jsonl'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                        records.append(record)
                    except Exception as e:
                        logger.warning(f"Failed to parse line {line_num} in {path}: {e}")
        except Exception as e:
            logger.error(f"Failed to open/read jsonl file {path}: {e}")
    else:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
            else:
                logger.warning(f"JSON file {path} did not contain a list or dict.")
        except Exception as e:
            logger.error(f"Failed to load json file {path}: {e}")
            
    return records

def load_single_candidate(record: dict) -> dict:
    """
    Validates that a raw dict has the required top-level keys.
    If any key is missing, fill with safe defaults:
      - missing list fields (career_history, education, skills,
        certifications, languages) -> []
      - missing dict fields (profile, redrob_signals) -> {}
      - missing candidate_id -> 'UNKNOWN'
    Never raises. Always returns something usable.
    """
    if not isinstance(record, dict):
        record = {}
        
    result = dict(record)
    
    # candidate_id
    if 'candidate_id' not in result or result['candidate_id'] is None:
        result['candidate_id'] = 'UNKNOWN'
    else:
        result['candidate_id'] = str(result['candidate_id'])
        
    # lists
    list_keys = ['career_history', 'education', 'skills', 'certifications', 'languages']
    for key in list_keys:
        if key not in result or result[key] is None or not isinstance(result[key], list):
            result[key] = []
            
    # dicts
    dict_keys = ['profile', 'redrob_signals']
    for key in dict_keys:
        if key not in result or result[key] is None or not isinstance(result[key], dict):
            result[key] = {}
            
    return result

def load_job_description(path: str) -> str:
    """
    Reads job_description.docx using python-docx.
    Joins all non-empty paragraph texts with newline.
    Returns full JD as a single string.
    """
    try:
        doc = docx.Document(path)
        paragraphs = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                paragraphs.append(text)
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Failed to load job description from {path}: {e}")
        return ""
