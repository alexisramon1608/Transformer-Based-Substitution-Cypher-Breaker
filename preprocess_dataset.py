from cipher_8bit import *
from french_dataset import get_full_dataset

def get_frequency_ranks(encodings, symbols, sequence_len):
    freq_ranks_dict = [0] * len(symbols)
    encodings = encodings[:sequence_len]
    for encoding in encodings:
        freq_ranks_dict[encoding] += 1
    freq_ranks = [0] * (sequence_len)

    for i in range(len(encodings)):
        freq_ranks[i] = freq_ranks_dict[encodings[i]]
    return freq_ranks

def get_proximity_array(encodings, sequence_len):
    distances = [0] * (sequence_len)
    encodings = encodings[:sequence_len]
    for i, encoding in enumerate(encodings):
        try:
            last_idx = encodings.index(encoding, 0, i)
            distances[i] = (i - last_idx)
        except ValueError:
            # If the encoding is not found in the indices, set the distance to 0
            distances[i] = 0
    return distances

def preprocess_text(sequence_len=256):
    full_text = get_full_dataset()
    symbols = get_symbols(full_text, 256)
    symbols = load_or_save_symbols(symbols)
    substitution_rule = substitution_cipher(symbols, 1337)

    i = 0
    raw_length = sequence_len * 2 # Overshoot so when it encodes it takes atleast sequence_len
    processed_data = []
    while i * raw_length < len(full_text) - raw_length:
        i += 1
        sample_text = full_text[(i - 1) * raw_length: i * raw_length - 1]
        encodings_array, indices = encode_text_with_indices(substitution_rule, symbols, sample_text)
        if len(encodings_array) > sequence_len: encodings_array = encodings_array[:sequence_len]
        if len(indices) > sequence_len: indices = indices[:sequence_len]

        ranks = get_frequency_ranks(encodings_array, symbols, sequence_len)
        distances = get_proximity_array(encodings_array, sequence_len)
        processed_data.append([encodings_array, distances, indices])
    return processed_data

