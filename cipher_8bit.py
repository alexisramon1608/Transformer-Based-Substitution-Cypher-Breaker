import os
import pickle
import random
from collections import Counter

def cut_string_into_pairs(text_corpus):
    pairs = []
    for i in range(0, len(text_corpus) - 1, 2):
        pairs.append(text_corpus[i:i + 2])
    if len(text_corpus) % 2 != 0:
        pairs.append(text_corpus[-1] + '_')
    return pairs

def get_symbols(text_corpus, max_characters=256):
    # Get all single unique characters, then fill in the rest of the symbol spots with the most common character pairs
    single_characters = list(set(list(text_corpus)))
    pairs = [item for item, _ in Counter(cut_string_into_pairs(text_corpus)).most_common(256 - len(single_characters))]
    return single_characters + pairs

def substitution_cipher(symbols, random_seed):
    random.seed(random_seed)
    # Make a randomly ordered range from 0-255 (8-bits)
    integer_encodings = random.sample(list(range(len(symbols))), len(symbols))
    substitution_rule = dict({})
    # Map every symbol to a unique encoding
    for idx, symbol in enumerate(symbols):
        encoding = integer_encodings[idx]  # Get the random encoding for the symbol
        substitution_rule[symbol] = encoding  # Store the symbol as the key, encoding as the value
    return substitution_rule

def encode_text_with_indices(rule, symbols, text):
    encoded_text = []
    indices = []
    i = 0

    # Create a reverse mapping from symbols to their indices
    index_dict = dict(zip(symbols, range(len(symbols))))

    while i < len(text):
        # Check for pairs
        if i + 1 < len(text):
            pair = text[i] + text[i + 1]
            # Check if the pair exists in the rule
            if pair in rule:
                encoding = rule[pair]  # Get the encoding for the pair
                encoded_text.append(encoding)
                indices.append(index_dict[pair])  # Get the index of the symbol
                i += 2  # Skip the two characters used in the pair
                continue

        # Single character substitution
        if text[i] in rule:
            encoding = rule[text[i]]
            encoded_text.append(encoding)
            indices.append(index_dict[text[i]])  # Get the index of the symbol
        else:
            # If the character doesn't exist in the rule, keep it as-is
            encoded_text.append(256)
            indices.append(256)  # Use -1 or some other value to indicate no encoding for this character

        i += 1

    return encoded_text, indices

# Function to load or save the symbols as a pickle
def load_or_save_symbols(symbols, pickle_file_path="symbols.pkl"):
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            print("Loading symbols from pickle file...")
            return pickle.load(f)
    else:
        print("Pickle file not found. Saving symbols...")
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(symbols, f)
        return symbols
