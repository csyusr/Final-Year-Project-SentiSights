import pandas as pd
from google import genai
import time
import json
import os
import re
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURATION ---
API_KEY = "Insert Api Key Here" # <--- PASTE KEY HERE

client = genai.Client(api_key=API_KEY)
# We keep using the high-intelligence model
MODEL_NAME = 'gemma-3-27b-it' 

def select_file(title="Select File"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("CSV Files", "*.csv")])

def clean_json_response(text):
    """
    Removes Markdown formatting (```json ... ```) so Python can read it.
    """
    # Remove code block markers
    clean_text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"```", "", clean_text)
    return clean_text.strip()

def extract_detailed_aspects(text):
    """
    Asks Gemma to extract granular details. 
    REMOVED 'response_mime_type' to fix the 400 Error.
    """
    prompt = (
        f"You are a precise data extraction tool. Analyze the hotel review below.\n"
        f"Identify every distinct opinion and extract these 4 fields:\n"
        f"1. Category: (Room, Service, Food, Location, Price, Amenities, Cleanliness)\n"
        f"2. Aspect_Term: The specific noun (e.g., 'bed', 'staff').\n"
        f"3. Opinion_Term: The specific adjective (e.g., 'hard', 'rude').\n"
        f"4. Sentiment: (Positive, Negative, Neutral)\n\n"
        
        f"--- REAL EXAMPLE ---\n"
        f"Input: \"Breakfast - perfect, everything so delicious. Just need extra 'sabar' for the replenishment due to PH.\"\n"
        f"Output JSON:\n"
        f"[\n"
        f"  {{\"Category\": \"Food\", \"Aspect_Term\": \"Breakfast\", \"Opinion_Term\": \"perfect\", \"Sentiment\": \"Positive\"}},\n"
        f"  {{\"Category\": \"Service\", \"Aspect_Term\": \"replenishment\", \"Opinion_Term\": \"need extra sabar\", \"Sentiment\": \"Negative\"}}\n"
        f"]\n"
        f"--------------------\n\n"

        f"Review: \"{text}\"\n"
        f"Output ONLY the JSON list. Do not write any introduction."
    )
    
    try:
        # --- FIXED CALL (Removed invalid config) ---
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        
        # Clean the text before parsing
        return clean_json_response(response.text)
        
    except Exception as e:
        print(f"  [!] API Error: {e}")
        time.sleep(5)
        return "[]"

def process_detailed_labeling():
    print("Step 1: Select your 'Refined' Dataset...")
    input_file = select_file("Select Dataset RC Refined.csv")
    if not input_file: return

    output_file = "Training_Data_Detailed_ABSA.csv"
    print(f"--- Processing {os.path.basename(input_file)} ---")
    
    df = pd.read_csv(input_file)
    
    if 'Translated_Text' not in df.columns:
        print("Error: 'Translated_Text' column missing.")
        return

    labeled_data = []
    
    # Resume Logic
    if os.path.exists(output_file):
        print("Found existing file. Resuming...")
        existing_df = pd.read_csv(output_file)
        processed_ids = existing_df['Original_Review_ID'].unique()
        labeled_data = existing_df.to_dict('records')
    else:
        processed_ids = []

    print("Starting Detailed Auto-Annotation...")
    counter = 0

    for index, row in df.iterrows():
        review_id = row['Review_ID']
        if review_id in processed_ids: continue

        review_text = row['Translated_Text']
        if not isinstance(review_text, str) or len(review_text) < 5: continue

        print(f"Processing ID {review_id}...", end=" ")
        
        json_str = extract_detailed_aspects(review_text)
        
        try:
            aspects = json.loads(json_str)
            if aspects:
                for item in aspects:
                    labeled_data.append({
                        'Original_Review_ID': review_id,
                        'Full_Review': review_text,
                        'Category': item.get('Category', 'Other'),
                        'Aspect_Term': item.get('Aspect_Term', ''),
                        'Opinion_Term': item.get('Opinion_Term', ''),
                        'Sentiment': item.get('Sentiment', 'Neutral')
                    })
                print(f"Found {len(aspects)} details.")
            else:
                print("None.")
            
        except json.JSONDecodeError:
            print(f"JSON Parse Error. (Raw: {json_str[:20]}...)")
            
        counter += 1
        if counter % 10 == 0:
            pd.DataFrame(labeled_data).to_csv(output_file, index=False)
            print(f"--> Saved {len(labeled_data)} rows.")
        
        time.sleep(2.0) 

    # Final Save
    pd.DataFrame(labeled_data).to_csv(output_file, index=False)
    print(f"\nSUCCESS! Saved to: {output_file}")

if __name__ == "__main__":
    process_detailed_labeling()