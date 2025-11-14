import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

print("Connecting to Google AI Studio...")

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
    else:
        genai.configure(api_key=api_key)

        print("\nFetching available Gemini models that support 'generateContent'...")

        count = 0
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"\n--- Found Model ---")
                print(f"Model name: {model.name}")
                print(f"   Description: {model.description}")
                count += 1

        if count == 0:
            print("\nNo models supporting 'generateContent' were found.")
        else:
            print(f"\nFound {count} usable models.")
            print("Please copy the 'Model name' (e.g., 'models/gemini-pro-vision') and paste it into your main.py file.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure your GEMINI_API_KEY is correct and you have an internet connection.")