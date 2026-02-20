import pandas as pd
import re

# 1. Load your current dataset
df = pd.read_csv("Final_Dataset_For_Website_Gemma.csv")

# 2. DROP MISSING VALUES (Crucial for modeling)
# We drop rows where Aspect_Term or Opinion_Term is missing (approx 70 rows)
df = df.dropna(subset=['Aspect_Term', 'Opinion_Term', 'Full_Review'])

# 3. TEXT CLEANING FUNCTION
def clean_text(text):
    text = str(text).lower()                 # Lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove punctuation/special chars
    return text

# Apply cleaning
df['Cleaned_Review'] = df['Full_Review'].apply(clean_text)

# 4. SOLVE THE "CONFUSED DATA" ISSUE
# We combine the Aspect with the Review so the model knows what to look for.
# Format: "aspect_term aspect_term ... full review text"
df['Model_Input'] = df['Aspect_Term'] + " " + df['Cleaned_Review']

# 5. LABEL ENCODING (Convert Sentiment to Numbers)
sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['Sentiment_Label'] = df['Sentiment'].map(sentiment_map)

# 6. Save the prepared file
df.to_csv("Ready_For_Modelling.csv", index=False)

print("Success! Created 'Ready_For_Modelling.csv'")
print("Use the 'Model_Input' column as your X (Features) and 'Sentiment_Label' as your y (Target).")