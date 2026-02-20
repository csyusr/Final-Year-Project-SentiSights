import pandas as pd
from google import genai
import time
import os
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURATION ---
# I have kept your key here so it runs immediately. 
# Be careful sharing this code with others!
API_KEY = "Insert Api Key Here" # <--- PASTE KEY HERE

client = genai.Client(api_key=API_KEY)
MODEL_NAME = 'gemma-3-27b-it' 

def select_file(title="Select File"):
    """
    Opens a window to select a CSV file.
    """
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("CSV Files", "*.csv")])

def get_gemma_recommendation(category, aspect, opinion):
    """
    Uses Gemma 27B to generate a strict 3-6 word action plan.
    """
    prompt = (
        f"Role: Hotel Operations Director.\n"
        f"Task: Give a specific, short action plan (3-6 words) to resolve this negative feedback.\n"
        f"Category: {category}\n"
        f"Complaint: The '{aspect}' was '{opinion}'.\n\n"
        f"Strict Constraint: Output ONLY the action plan. Do not use full sentences.\n"
        f"Action Plan:"
    )
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        # Clean up text (remove markdown, newlines, or "Review:" labels)
        text = response.text.strip()
        text = text.replace("**", "").replace("Action Plan:", "").strip()
        return text
    except Exception as e:
        print(f"  [!] API Error: {e}")
        time.sleep(2)
        return "Review Standard Operating Procedure"

def main():
    # 1. Ask user for the file (Select the PARTIAL file to resume)
    print("Step 1: Select 'Final_Dataset_For_Website_Gemma.csv' (The partial file to resume)...")
    input_file = select_file("Select File to Resume")
    if not input_file: 
        print("No file selected. Exiting.")
        return

    output_file = "Final_Dataset_For_Website_Gemma.csv"
    print(f"--- Resuming Processing on {os.path.basename(input_file)} ---")
    
    df = pd.read_csv(input_file)
    
    # 2. Create the column if it's missing (First run safety)
    if 'Recommendation' not in df.columns:
        df['Recommendation'] = ""

    # 3. Calculate Totals
    # We filter specifically for "Negative" sentiment to know the total work needed.
    negatives = df[df['Sentiment'] == 'Negative']
    total_neg = len(negatives)
    
    print(f"Total Negative Reviews found in file: {total_neg}")
    print(f"Model: {MODEL_NAME}")
    print("-" * 50)

    # 4. Initialize Counter
    # This tracks which negative review number we are currently on.
    current_neg_count = 0 
    
    for index, row in df.iterrows():
        sentiment = str(row['Sentiment'])
        
        # --- PROCESS ONLY NEGATIVE REVIEWS ---
        if sentiment == 'Negative':
            # Increment the counter immediately because we found a negative review
            current_neg_count += 1
            
            # --- RESUME LOGIC ---
            # Check if this row ALREADY has a recommendation
            current_rec = str(row['Recommendation']).strip()
            
            # If it has text and is not "nan", it is finished. SKIP IT.
            if current_rec and current_rec.lower() != "nan": 
                # We skip, but 'current_neg_count' stays accurate so the next print is correct.
                continue

            # --- IF WE GET HERE, THE ROW IS EMPTY. PROCESS IT. ---
            print(f"[{current_neg_count}/{total_neg}] Issue: {row['Aspect_Term']} ({row['Opinion_Term']})...", end=" ")
            
            rec = get_gemma_recommendation(row['Category'], row['Aspect_Term'], row['Opinion_Term'])
            
            # Save to DataFrame memory
            df.at[index, 'Recommendation'] = rec
            print(f"-> {rec}")
            
            # Sleep to prevent API Overload
            time.sleep(1.0)
            
            # Save to Disk every 10 rows (Safety Save)
            if index % 10 == 0:
                df.to_csv(output_file, index=False)
        
        else:
            # Ensure Positive/Neutral/Empty rows stay empty
            df.at[index, 'Recommendation'] = ""

    # 5. Final Save when loop finishes
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSUCCESS! All {total_neg} reviews processed.")
    print(f"Saved final file to: {output_file}")

if __name__ == "__main__":
    main()