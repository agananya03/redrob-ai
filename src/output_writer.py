import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def write_submission(ranked_df: pd.DataFrame,
                     output_path: str = 'outputs/ranked_candidates.csv'
                     ) -> None:
    """
    Writes the ranked DataFrame to a CSV file matching the required submission format.
    
    Args:
        ranked_df (pd.DataFrame): The DataFrame containing ranked candidates.
        output_path (str, optional): The path to save the output CSV. Defaults to 'outputs/ranked_candidates.csv'.
        
    Raises:
        ValueError: If ranked_df is empty.
    """
    if ranked_df.empty:
        raise ValueError("Cannot write submission: ranked_df is empty.")
        
    df = ranked_df.copy()
    
    # 1. 'score' = calibrated_score column, rounded to 4 decimal places
    if 'calibrated_score' in df.columns:
        df['score'] = df['calibrated_score'].round(4)
    elif 'final_score' in df.columns:
        df['score'] = df['final_score'].round(4)
    else:
        df['score'] = 0.0
    
    # 2. 'rank' = integer, 1-indexed.
    if 'rank' not in df.columns:
        df['rank'] = range(1, len(df) + 1)
    df['rank'] = df['rank'].astype(int)
    
    # 3. 'reasoning' = string column. If doesn't exist, fill with empty string.
    if 'reasoning' not in df.columns:
        df['reasoning'] = ""
    else:
        df['reasoning'] = df['reasoning'].fillna("").astype(str)
        
    # Filter and reorder columns
    cols = ['candidate_id', 'rank', 'score', 'reasoning']
    df = df[cols]
    
    # 4. Create outputs/ directory if it doesn't exist.
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    # 5. Output must have header row, no index
    df.to_csv(output_path, index=False)
    
    logger.info(f"Written {len(df)} candidates to {output_path}")
