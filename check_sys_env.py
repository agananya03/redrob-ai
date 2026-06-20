import os

api_key = os.environ.get('GROQ_API_KEY')
if api_key:
    print(f"System ENV Key found. Starts with: {api_key[:5]}... Length: {len(api_key)}")
else:
    print("No system ENV key found.")
