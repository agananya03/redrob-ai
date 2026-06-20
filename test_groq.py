import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

if not api_key:
    print("No API key found.")
else:
    print(f"API Key found. Starts with: {api_key[:5]}... Length: {len(api_key)}")
    if api_key.startswith('"') or api_key.startswith("'"):
        print("WARNING: Key has quotes around it!")
        
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': 'Say hi'}],
            max_tokens=10
        )
        print("API Call SUCCESS!")
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("API Call FAILED.")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
