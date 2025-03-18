import pandas as pd
import re
from langdetect import detect, DetectorFactory
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure reproducible language detection results
DetectorFactory.seed = 0

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt_tab')
# Define a function to check if a text is in English.
def is_english(text):
    try:
        return detect(text) == 'en'
    except Exception:
        return False

# Function to clean each review text.
def clean_text(text):
    # Remove newline characters and extra spaces.
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(' +', ' ', text)
    
    # Tokenize the text. This will keep punctuation and emojis as tokens.
    tokens = word_tokenize(text)
    
    # Prepare English stopwords list.
    stop_words = set(stopwords.words('english'))
    
    # Filter out stopwords. Note: emojis are not part of stopwords.
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Rejoin tokens into cleaned text.
    cleaned = " ".join(filtered_tokens)
    return cleaned

def main():
    # Load the CSV file with your Steam reviews.
    input_file = 'steam_reviews.csv'
    df = pd.read_csv(input_file)
    
    # Drop rows with empty reviews.
    df = df.dropna(subset=['review'])
    
    # Filter reviews to keep only English texts.
    df['is_english'] = df['review'].apply(is_english)
    df = df[df['is_english']].copy()
    
    # Clean up the review text.
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Save the cleaned data to a new CSV file.
    output_file = 'steam_reviews_cleaned.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    main()
