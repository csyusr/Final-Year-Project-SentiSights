import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

def select_file(title="Select File"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("CSV Files", "*.csv")])

def main():
    # 1. Load the ABSA Output (The "Second File")
    print("Step 1: Select 'Training_Data_Detailed_ABSA.csv'...")
    absa_file = select_file("Select ABSA Output File")
    if not absa_file: return
    
    # 2. Load the Original Dataset (The "Third File")
    print("Step 2: Select 'Dataset RC Refined.csv'...")
    orig_file = select_file("Select Original Dataset")
    if not orig_file: return

    print(f"Merging files...")
    
    df_absa = pd.read_csv(absa_file)
    df_orig = pd.read_csv(orig_file)

    # 3. The Merge Operation
    # We take specific columns from the Original file and attach them to the ABSA file
    # Key: 'Original_Review_ID' (ABSA) <--> 'Review_ID' (Original)
    
    columns_to_add = ['Review_ID', 'Reviewer_Name', 'Date', 'Branch_Name', 'Rating']
    
    merged_df = pd.merge(
        df_absa,
        df_orig[columns_to_add],
        left_on='Original_Review_ID',
        right_on='Review_ID',
        how='left' # Keep all ABSA rows, just add info where matches found
    )

    # 4. Clean Up Columns
    # We might have duplicate ID columns now, let's organize them nicely
    # Desired Order: ID, Date, Branch, Reviewer, Rating, Category, Aspect, Opinion, Sentiment
    
    final_columns = [
        'Original_Review_ID', 
        'Date', 
        'Branch_Name', 
        'Reviewer_Name', 
        'Rating',
        'Full_Review',
        'Category', 
        'Aspect_Term', 
        'Opinion_Term', 
        'Sentiment'
    ]
    
    # Filter to ensure we only select columns that actually exist
    final_columns = [c for c in final_columns if c in merged_df.columns]
    merged_df = merged_df[final_columns]

    # 5. Save
    output_file = "Training_Data_With_Metadata.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\nSUCCESS! Created: {output_file}")
    print(f"Total Rows: {len(merged_df)}")
    print("\n--- Preview ---")
    print(merged_df.head())

if __name__ == "__main__":
    main()