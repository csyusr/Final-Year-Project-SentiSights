import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

def select_file(title="Select File"):
    """Opens a pop-up to select a file."""
    root = tk.Tk()
    root.withdraw() # Hide the main window
    file_path = filedialog.askopenfilename(title=title, filetypes=[("CSV Files", "*.csv")])
    return file_path

def save_file(title="Save File"):
    """Opens a pop-up to save a file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(title=title, defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    return file_path

def clean_dataset():
    # 1. Select Input File
    print("Please select your source CSV file...")
    input_file = select_file("Select your Source Dataset (CSV)")
    
    if not input_file:
        print("No file selected. Exiting.")
        return

    print(f"--- Processing {os.path.basename(input_file)} ---")
    
    # 2. Load Data
    try:
        df = pd.read_csv(input_file)
        print(f"Initial row count: {len(df)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 3. Remove Missing Data
    df_clean = df.dropna(subset=['text']).copy()
    print(f"Rows after removing empty reviews: {len(df_clean)}")

    # 4. Date Formatting
    if 'publishedAtDate' in df_clean.columns:
        df_clean['publishedAtDate'] = pd.to_datetime(df_clean['publishedAtDate'])
        df_clean['Date'] = df_clean['publishedAtDate'].dt.date

    # 5. Column Renaming
    column_mapping = {
        'text': 'Review_Text',
        'title': 'Branch_Name',
        'stars': 'Rating',
        'name': 'Reviewer_Name'
    }
    df_clean = df_clean.rename(columns=column_mapping)

    # 6. ID Creation
    df_clean.insert(0, 'Review_ID', range(1, len(df_clean) + 1))

    # 7. Column Selection (UPDATED ORDER)
    # Review_ID -> Reviewer_Name -> Date -> Branch_Name -> Rating -> Review_Text
    final_columns = ['Review_ID', 'Reviewer_Name', 'Date', 'Branch_Name', 'Rating', 'Review_Text']
    
    # Only keep columns that actually exist
    available_columns = [c for c in final_columns if c in df_clean.columns]
    df_final = df_clean[available_columns]

    # 8. Save Output
    print("Please choose where to save the cleaned file...")
    output_file = save_file("Save Cleaned Dataset As...")
    
    if output_file:
        # encoding='utf-8-sig' ensures Chinese/Malay characters work in Excel
        df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"--- SUCCESS! Saved cleaned data to: {output_file} ---")
        print("\nPreview of cleaned data:")
        print(df_final.head())
    else:
        print("Save cancelled.")

if __name__ == "__main__":
    clean_dataset()