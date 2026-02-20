import pandas as pd
import spacy
import re
import tkinter as tk
from tkinter import filedialog
import os

# Load the English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    exit()

def select_file(title="Select File"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("CSV Files", "*.csv")])

def save_file(title="Save File"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.asksaveasfilename(title=title, defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])

def basic_clean(text):
    """
    Removes heavy noise like newlines, special symbols, and extra spaces.
    """
    if not isinstance(text, str): return ""
    
    # 1. Remove newlines and tabs
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # 2. Remove weird scraper artifacts like '||'
    text = text.replace('||', ' ')
    
    # 3. Remove multiple spaces
    text = re.sub(' +', ' ', text)
    
    # 4. Lowercase
    return text.strip().lower()

def extract_important_words(text):
    """
    Keeps ONLY Nouns, Adjectives, Adverbs, and Negations.
    Removes everything else (verbs, determiners, etc).
    """
    doc = nlp(text)
    
    keep_tags = ['NOUN', 'PROPN', 'ADJ', 'ADV'] # Noun, Proper Noun, Adjective, Adverb
    important_words = []
    
    for token in doc:
        # Keep word if it is in our important list OR if it is a negation (like 'not', 'no')
        if token.pos_ in keep_tags or token.dep_ == 'neg':
            # Lemmatization: Converts "rooms" -> "room", "cleaning" -> "clean"
            important_words.append(token.lemma_)
            
    return " ".join(important_words)

def process_refinement():
    # 1. Select File
    print("Select your 'Translated' dataset...")
    input_file = select_file("Select Translated Dataset")
    if not input_file: return

    print(f"--- Loading {os.path.basename(input_file)} ---")
    df = pd.read_csv(input_file)
    
    # Ensure Translated_Text exists
    if 'Translated_Text' not in df.columns:
        print("Error: Column 'Translated_Text' not found!")
        return

    # 2. Apply Basic Cleaning (Good for CNN/Deep Learning)
    print("Step 1: Performing Basic Cleaning (Removing noise)...")
    df['Text_Cleaned_Full'] = df['Translated_Text'].apply(basic_clean)

    # 3. Apply Advanced Extraction (Good for Naive Bayes / SVM)
    print("Step 2: Extracting Aspects & Sentiments (Removing filler)...")
    # This creates the "Noun + Adjective" only version you asked for
    df['Text_Refined_Keywords'] = df['Text_Cleaned_Full'].apply(extract_important_words)

    # 4. Save
    print("Select where to save the refined dataset...")
    output_file = save_file("Save Refined Dataset As...")
    
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSUCCESS! Saved to: {output_file}")
        print("\n--- Example Comparison ---")
        print(df[['Translated_Text', 'Text_Refined_Keywords']].head(3))

if __name__ == "__main__":
    process_refinement()