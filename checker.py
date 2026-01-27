import os
import pandas as pd
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load API key from .env file
load_dotenv()

# --- CONFIGURATION ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Using Flash because it's fast and cheap for high volume
model = genai.GenerativeModel('gemini-1.5-flash') 

PATHWAY_URL = "http://127.0.0.1:8000/v1/retrieve"
INPUT_CSV = "./data/test.csv"     # This must match your CSV file location
OUTPUT_CSV = "./results.csv"      # The file to submit

# --- HELPER 1: Extract Claims ---
def get_claims_from_backstory(backstory_text):
    """Asks Gemini to extract factual claims from the backstory."""
    prompt = f"""
    Analyze this character backstory. Extract 3-5 specific, distinct factual claims 
    (e.g. childhood events, specific fears, family details, physical traits).
    Return ONLY a Python list of strings. Example: ["He was born in Ohio", "He hates dogs"]
    
    Backstory:
    {backstory_text}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip().replace("```python", "").replace("```", "")
        return eval(text)
    except:
        return []

# --- HELPER 2: Get Evidence ---
def query_pathway(query_text, k=3):
    """Ask Pathway server for relevant paragraphs from the novels."""
    payload = {"query": query_text, "k": k}
    try:
        response = requests.post(PATHWAY_URL, json=payload)
        if response.status_code == 200:
            results = response.json()
            return [item['text'] for item in results]
        return []
    except:
        return []

# --- HELPER 3: The Judge ---
def judge_consistency(claim, evidence_texts):
    """Ask Gemini if the evidence contradicts the claim."""
    if not evidence_texts: return "NEUTRAL"
    
    evidence_block = "\n---\n".join(evidence_texts)
    prompt = f"""
    You are a continuity editor.
    CLAIM: "{claim}"
    EVIDENCE FROM NOVEL:
    {evidence_block}
    
    Task: Does the evidence CONTRADICT the claim? 
    - Output 'CONTRADICTION' only if facts are strictly impossible together.
    - Output 'CONSISTENT' if it fits or is not mentioned.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.upper()
    except:
        return "CONSISTENT"

# --- MAIN LOGIC ---
def process_csv():
    print(f"📂 Reading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("❌ Error: test.csv not found in data folder!")
        return

    predictions = []
    rationales = []
    
    for index, row in df.iterrows():
        # Handle column names (adjust if your CSV headers are different)
        story_id = row.get('story_id', index)
        backstory = row.get('backstory', row.get('Backstory', ''))
        
        print(f"\nProcessing Story {story_id}...")
        
        # 1. Break backstory into claims
        claims = get_claims_from_backstory(backstory)
        
        is_consistent = 1
        conflict_reason = "Consistent"
        
        # 2. Check each claim
        if claims:
            for claim in claims:
                evidence = query_pathway(claim)
                verdict = judge_consistency(claim, evidence)
                
                if "CONTRADICTION" in verdict:
                    is_consistent = 0
                    conflict_reason = f"Conflict: {claim}"
                    print(f"  ❌ {conflict_reason}")
                    break # Stop checking this story if we found a lie
        else:
            print("  ⚠️ No claims extracted.")

        predictions.append(is_consistent)
        rationales.append(conflict_reason)
        
        time.sleep(1) # Be nice to the API

    # 3. Save
    df['prediction'] = predictions
    df['rationale'] = rationales
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_csv()