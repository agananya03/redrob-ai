with open('src/structured_scoring.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Conflict 1
marker1_start = content.find('<<<<<<< HEAD\n        self.esco_extractor = None')
marker1_end = content.find('>>>>>>> 79d44c2b9ca0aaf27b5857efd47418ae85fd395d\n', marker1_start)
if marker1_start != -1 and marker1_end != -1:
    before = content[:marker1_start]
    after = content[marker1_end + len('>>>>>>> 79d44c2b9ca0aaf27b5857efd47418ae85fd395d\n'):]
    replacement1 = '''        self.esco_extractor = None
        self.esco_failed = False
        self._jd_cache = {}
        # Precompile regular expressions for efficiency
        import re
        self.degree_ms_re = re.compile(r'\\\\bms\\\\b|\\\\bm\\\\.s\\\\b', re.IGNORECASE)
        self.degree_be_re = re.compile(r'\\\\bbe\\\\b|\\\\bbs\\\\b', re.IGNORECASE)
        self.field_cs_re = re.compile(r'\\\\bcs\\\\b|\\\\bai\\\\b|\\\\bml\\\\b', re.IGNORECASE)
        self.it_re = re.compile(r'\\\\bit\\\\b', re.IGNORECASE)

    def _get_jd_esco_skills(self, jd_text: str) -> set:
        cache_path = 'data/processed/jd_esco_skills.json'
        
        # 1. Try cache
        import os
        if os.path.exists(cache_path):
            try:
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load JD ESCO skills from cache: {e}")
                
        # 2. Extract
        if self.esco_failed:
            return set()
            
        if self.esco_extractor is None:
            try:
                from esco_skill_extractor import SkillExtractor
                self.esco_extractor = SkillExtractor()
            except Exception as e:
                logger.warning(f"Failed to initialize esco_skill_extractor: {e}")
                self.esco_failed = True
                return set()
                
        try:
            # get_skills expects List[str]
            jd_skills_lists = self.esco_extractor.get_skills([jd_text])
            if jd_skills_lists and len(jd_skills_lists) > 0:
                jd_skills = set(jd_skills_lists[0])
            else:
                jd_skills = set()
                
            # Cache it
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            import json
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(list(jd_skills), f)
                
            return jd_skills
        except Exception as e:
            logger.warning(f"Failed to extract JD ESCO skills: {e}")
            self.esco_failed = True
            return set()

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
        import re
        for term in known_tech_terms:
            escaped = re.escape(term)
            if re.search(r'\\\\b' + escaped + r'\\\\b', jd_text_lower):
                jd_skills.add(term)
            elif term in ['c++', 'c#'] and term in jd_text_lower:
                jd_skills.add(term)

        # Parse experience requirement from jd_text
        match_range = re.search(r'(\\\\d+)\\\\s*-\\\\s*(\\\\d+)\\\\s*years', jd_text, re.IGNORECASE)
        match_plus = re.search(r'(\\\\d+)\\\\+\\\\s*years', jd_text, re.IGNORECASE)
        match_at_least = re.search(r'at least\\\\s+(\\\\d+)\\\\s*years', jd_text, re.IGNORECASE)

        min_required = 5
        if match_range:
            min_required = int(match_range.group(1))
        elif match_plus:
            min_required = int(match_plus.group(1))
        elif match_at_least:
            min_required = int(match_at_least.group(1))

        info = (jd_skills, min_required)
        self._jd_cache[jd_text] = info
        return info\n'''.replace('\\\\', '\\')
    content = before + replacement1 + after

# Fix Conflict 2
marker2_start = content.find('<<<<<<< HEAD\n        jd_esco_skills = self._get_jd_esco_skills(jd_text)\n        \n=======\n>>>>>>> 79d44c2b9ca0aaf27b5857efd47418ae85fd395d')
if marker2_start != -1:
    before = content[:marker2_start]
    after = content[marker2_start + len('<<<<<<< HEAD\n        jd_esco_skills = self._get_jd_esco_skills(jd_text)\n        \n=======\n>>>>>>> 79d44c2b9ca0aaf27b5857efd47418ae85fd395d'):]
    replacement2 = '''        jd_esco_skills = self._get_jd_esco_skills(jd_text)'''
    content = before + replacement2 + after

# Remove fallback block
import re
fallback_pattern = re.compile(r'# Known tech terms to scan case-insensitively in jd_text.*?jd_skills\.add\(term\)', re.DOTALL)
content = fallback_pattern.sub('', content)

with open('src/structured_scoring.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Merge conflicts resolved.")
