"""
data_loader.py

Responsible for loading the raw dataset and job description.
"""

import os
import json
import logging
import docx

logger = logging.getLogger(__name__)

def load_candidates(path: str) -> list[dict]:
    """
    Reads candidate data from either a .jsonl or .json file.
    
    Args:
        path (str): The file path to load candidates from.
        
    Returns:
        list[dict]: A list of raw candidate dictionaries.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    path_lower = path.lower()
    if not os.path.exists(path):
        if path_lower.endswith('.jsonl'):
            raise FileNotFoundError(f"candidates.jsonl not found at {path}")
        else:
            raise FileNotFoundError(f"File not found at {path}")
            
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
    Validates a raw candidate dict and fills missing keys with safe defaults.
    
    Args:
        record (dict): A raw candidate dictionary.
        
    Returns:
        dict: A validated candidate dictionary with default values for missing keys.
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
    Reads a job description document (.docx) and returns its text.
    
    Args:
        path (str): The file path to the job description document.
        
    Returns:
        str: The full job description text as a single string.
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
