import os
from groq import Groq

api_key = os.getenv('GROQ_API_KEY')
print(f"API key available: {bool(api_key)}")

try:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'system', 'content': 'You are an expert.'},
            {'role': 'user', 'content': 'Say hi.'}
        ],
        max_tokens=80,
        temperature=0.3,
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    import traceback
    traceback.print_exc()
