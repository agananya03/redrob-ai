import pandas as pd

df = pd.DataFrame({
    'rank': [1],
    'current_title': ['Software Engineer'],
    'candidate_id': ['C123'],
    'final_score': [0.95],
    'skill_match_score': [0.9],
    'experience_score': [0.8],
    'platform_signal_score': [0.7],
    'reasoning': ['Good candidate']
})

bg_col = 'rgba(255,255,255,0.03)'
bord_col = 'rgba(255,255,255,0.06)'

table_html = f"<div class='candidate-card' style='padding: 0; overflow: hidden; color: white;'><table style='width: 100%; border-collapse: collapse; text-align: left;'>"
table_html += f"""
<tr style='background: {bg_col}; border-bottom: 1px solid {bord_col};'>
    <th style='padding: 1rem;'>Rank</th>
    <th style='padding: 1rem;'>Candidate</th>
    <th style='padding: 1rem;'>Final Score</th>
    <th style='padding: 1rem;'>Skills</th>
    <th style='padding: 1rem;'>Exp</th>
    <th style='padding: 1rem;'>Platform</th>
</tr>
"""
for idx, row in df.iterrows():
    table_html += f"""
    <tr style='border-bottom: 1px solid {bord_col};'>
        <td style='padding: 1rem;'><b>#{row['rank']}</b></td>
        <td style='padding: 1rem;'><b>{row['current_title']}</b> <br><span style='font-size:0.85rem; opacity:0.7;'>ID: {row['candidate_id']}</span></td>
        <td style='padding: 1rem;'><b style='color: #e56b40; font-size: 1.1rem;'>{row['final_score']:.3f}</b></td>
        <td style='padding: 1rem;'>{row['skill_match_score']:.2f}</td>
        <td style='padding: 1rem;'>{row['experience_score']:.2f}</td>
        <td style='padding: 1rem;'>{row['platform_signal_score']:.2f}</td>
    </tr>
    """
    if row['reasoning']:
        table_html += f"""
        <tr style='border-bottom: 1px solid {bord_col};'>
            <td colspan='6' style='padding: 0.75rem 1rem; background: {bg_col}; font-style: italic; font-size: 0.9rem;'>
                <span style='color: #e56b40; font-weight: bold;'>AI Reasoning:</span> {row['reasoning']}
            </td>
        </tr>
        """
table_html += "</table></div>"

with open('test.html', 'w') as f:
    f.write(table_html)
