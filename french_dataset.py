import re
import os
import requests

def clean_text(combined_text):
    # Regex pattern to keep only English/French letters, punctuation, and newlines
    allowed_chars_pattern = r"[A-Za-zÀ-ÿéèêàâçîôùûïëô\s\.,;!?()\"'\-\n]"
    
    # Keep only the characters that match the allowed pattern
    cleaned_text = ''.join(re.findall(allowed_chars_pattern, combined_text))
    
    return cleaned_text

def fetch_gutenberg_text(book_id):
    url = f"http://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch book ID {book_id}")
        return ""

def get_full_dataset(file_path: str = "full_dataset.txt") -> str:
    # List of French text IDs from Project Gutenberg
    french_text_ids = [
        20705,68719,68010,27837,28604,28605,4772,20372,12230,12782,12893,13192,42064,51237,67102,61627,64065,71553,57420,68501,67924,69597,68327,70061,70354,35986,48683,17319,14115,16240,74080,33033,74455,64663,74398,51372,71272,71093,73884,68865,72808,71143,64913,68199,67719,42064,51515,55072,70161,14071
    ]

    # Check if the file exists
    if os.path.exists(file_path):
        # Load and return the existing text file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    # Initialize an empty string to hold all the text
    combined_text = ""

    # Fetch and append each text
    for text_id in french_text_ids:
        text = fetch_gutenberg_text(text_id)
        if text:
            combined_text += text + "\n\n"  # Separate texts with double newlines

    # Clean the text before returning
    full_text = clean_text(combined_text)

    # Save the combined text to a file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(full_text)

    return full_text