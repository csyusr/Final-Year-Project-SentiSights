import pandas as pd
from google import genai # <--- NEW LIBRARY IMPORT
import time
import os
import tkinter as tk
from tkinter import filedialog
from langdetect import detect, LangDetectException

# --- CONFIGURATION ---
# PASTE YOUR API KEY HERE
API_KEY = "Insert Api Key Here"

# --- NEW CLIENT SETUP ---
# Instead of genai.configure, we create a Client
client = genai.Client(api_key=API_KEY)

# The model name you requested
MODEL_NAME = 'gemma-3-27b-it' 

def select_file(title="Select File"):
    """Opens a pop-up to select a file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("CSV Files", "*.csv")])
    return file_path

def save_file(title="Save File"):
    """Opens a pop-up to save a file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(title=title, defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    return file_path

def is_likely_english(text):
    """
    Returns True if text is likely English (to skip translation).
    """
    if not isinstance(text, str) or len(text) < 3:
        return True # Skip empty/short
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return True 

def translate_row(text):
    """
    Sends text to Gemma via the NEW Client.
    """
    prompt = (
        f"Translate the following hotel review into clear, standard English. "
        f"Accurately capture the sentiment and meaning of any slang or mixed languages. "
        f"Output ONLY the translated text.\n\n"
        f"Review: {text}"
    )
    try:
        # --- NEW CALL SYNTAX ---
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        
        # Check if we got a valid text response
        if response.text:
            return response.text.strip()
        else:
            return text
            
    except Exception as e:
        print(f"  [!] API Error: {e}")
        time.sleep(10) # Wait longer if error occurs
        return text

def process_translation():
    # 1. Select Input File
    print("Step 1: Select your Cleaned Dataset...")
    input_file = select_file("Select Source Dataset (Cleaned CSV)")
    if not input_file: return

    # 2. Select Output File
    print("Step 2: Where do you want to save the translations?")
    output_file = save_file("Save Translated Dataset As...")
    if not output_file: return

    print(f"--- Processing {os.path.basename(input_file)} ---")

    # 3. SMART LOAD (Resume Logic)
    if os.path.exists(output_file):
        print(f"Found existing file: {os.path.basename(output_file)}")
        print("Resuming from where you left off...")
        try:
            df = pd.read_csv(output_file)
            
            # Merge with original input to ensure we have all rows
            # (In case the output file was partial)
            df_source = pd.read_csv(input_file)
            
            # We use 'Review_ID' to align them if possible, otherwise index
            # This 'combine_first' fills in missing rows in 'df' using 'df_source'
            if len(df) < len(df_source):
                print(f"Merging... (Source: {len(df_source)}, Progress: {len(df)})")
                # Ensure we have the structure to merge
                if 'Translated_Text' not in df.columns:
                    df['Translated_Text'] = df['Review_Text']
                
                # Re-index to ensure alignment if Review_ID exists
                if 'Review_ID' in df.columns and 'Review_ID' in df_source.columns:
                    df = df.set_index('Review_ID')
                    df_source = df_source.set_index('Review_ID')
                    df = df.combine_first(df_source)
                    df = df.reset_index()
                else:
                    df = df_source.combine_first(df)

        except Exception as e:
            print(f"Error reading resume file ({e}). Starting fresh.")
            df = pd.read_csv(input_file)
            df['Translated_Text'] = df['Review_Text']
    else:
        print("Starting fresh...")
        df = pd.read_csv(input_file)
        # Create column for translation (copy original first)
        df['Translated_Text'] = df['Review_Text']

    # 4. Processing Loop
    counter = 0
    total_translated_session = 0
    
    print(f"Starting Translation with {MODEL_NAME} (New Client)...")
    print("Press Ctrl+C to stop at any time (progress is saved automatically).")

    for index, row in df.iterrows():
        original_text = row['Review_Text']
        current_translation = row['Translated_Text']

        # --- SMART SKIP ---
        # If translation is already different from original, it's done.
        if isinstance(current_translation, str) and isinstance(original_text, str):
            if current_translation != original_text:
                continue 

        # 1. Language Check
        if is_likely_english(original_text):
            continue 

        # 2. Translate
        print(f"Translating Row {index}...", end=" ")
        
        new_text = translate_row(original_text)
        
        if new_text != original_text:
            print("Done.")
            df.at[index, 'Translated_Text'] = new_text
            total_translated_session += 1
        else:
            print("Skipped (Error/Same).")

        counter += 1

        # 3. Save Progress (Every 5 rows)
        if counter % 5 == 0:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"--> Saved progress ({total_translated_session} new translations)")
        
        # 4. Pause (Safety for 27B model)
        time.sleep(4.0) 

    # Final Save
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSUCCESS! Session finished.")
    print(f"Total rows translated this session: {total_translated_session}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    process_translation()