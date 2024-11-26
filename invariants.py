from cipher_8bit import *
from french_dataset import get_full_dataset
import json

def get_pattern_ranks(pattern_frequency_dict):
    sorted_items = sorted(pattern_frequency_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize a new dictionary for ranks
    ranked_dict = {}
    
    # Assign ranks (starting from 1)
    rank = 1
    for key, value in sorted_items:
        ranked_dict[key] = rank
        rank += 1
    return ranked_dict

def unique_pattern_identifiers(symbol_index_sequences,save=False, name="data"):
    freq_dict = {}
    i = 0
    if os.path.exists(name+".json") and save:
        # Load the existing JSON data into a dictionary
        with open(name+".json", "r") as json_file:
            loaded_dict = json.load(json_file)
        return get_pattern_ranks(loaded_dict)

    for sequence in symbol_index_sequences:
        raw_pattern_data = find_patterns_and_indices(sequence, remove_subsets=False)
        for pattern in raw_pattern_data:
            key = "-".join(map(str, pattern[0]))
            if key in freq_dict:
                freq_dict[key] += len(pattern[1])
            else:
                freq_dict[key] = len(pattern[1])
        i += 1
    if save:
        with open(name+".json", "w") as json_file:
            json.dump(freq_dict, json_file, indent=4)

    return get_pattern_ranks(freq_dict)

def get_data_pairs(full_text):
    text_chunks = []
    chunk_len = 3000
    i=0
    while i * chunk_len < len(full_text) - chunk_len - 1:
        i += 1
        sample_text = full_text[(i - 1) * chunk_len: i * chunk_len - 1]
        text_chunks.append(sample_text)
    symbol_index_sequences = []
    symbols =  symboles = ['b', 'j', '\r', 'J', '”', ')', 'Â', 'É', 'ê', '5', 't', '9', 'Y', '%', 'N', 'B', 'V', '\ufeff', 'Ê', '?', '’', 'i', ':', 's', 'C', 'â', 'ï', 'W', 'y', 'p', 'D', '—', '«', 'º', 'A', '3', 'n', '0', 'q', '4', 'e', 'T', 'È', '$', 'U', 'v', '»', 'l', 'P', 'X', 'Z', 'À', 'ç', 'u', '…', 'î', 'L', 'k', 'E', 'R', '2', '_', '8', 'é', 'O', 'Î', '‘', 'a', 'F', 'H', 'c', '[', '(', "'", 'è', 'I', '/', '!', ' ', '°', 'S', '•', '#', 'x', 'à', 'g', '*', 'Q', 'w', '1', 'û', '7', 'G', 'm', '™', 'K', 'z', '\n', 'o', 'ù', ',', 'r', ']', '.', 'M', 'Ç', '“', 'h', '-', 'f', 'ë', '6', ';', 'd', 'ô', 'e ', 's ', 't ', 'es', ' d', '\r\n', 'en', 'qu', ' l', 're', ' p', 'de', 'le', 'nt', 'on', ' c', ', ', ' e', 'ou', ' q', ' s', 'n ', 'ue', 'an', 'te', ' a', 'ai', 'se', 'it', 'me', 'is', 'oi', 'r ', 'er', ' m', 'ce', 'ne', 'et', 'in', 'ns', ' n', 'ur', 'i ', 'a ', 'eu', 'co', 'tr', 'la', 'ar', 'ie', 'ui', 'us', 'ut', 'il', ' t', 'pa', 'au', 'el', 'ti', 'st', 'un', 'em', 'ra', 'e,', 'so', 'or', 'l ', ' f', 'll', 'nd', ' j', 'si', 'ir', 'e\r', 'ss', 'u ', 'po', 'ro', 'ri', 'pr', 's,', 'ma', ' v', ' i', 'di', ' r', 'vo', 'pe', 'to', 'ch', '. ', 've', 'nc', 'om', ' o', 'je', 'no', 'rt', 'à ', 'lu', "'e", 'mo', 'ta', 'as', 'at', 'io', 's\r', 'sa', "u'", 'av', 'os', ' à', ' u', "l'", "'a", 'rs', 'pl', 'é ', '; ', 'ho', 'té', 'ét', 'fa', 'da', 'li', 'su', 't\r', 'ée', 'ré', 'dé', 'ec', 'nn', 'mm', "'i", 'ca', 'uv', '\n\r', 'id', ' b', 'ni', 'bl']
    symbols = load_or_save_symbols(symbols)
    substitution_rule = substitution_cipher(symbols, 1337)

    def invariate_sequence(sample, ids, vocab_size):
        fill_in = []
        p = find_patterns_and_indices(sample)
        u = find_unique_singles(sample)
        for pattern in p:
            value = "-".join(map(str, pattern[0]))
            for index in pattern[1]:
                fill_in.append([index, ids[value], len(pattern[0])])
        for unique in u:
            fill_in.append([unique[1][0], vocab_size + 1, 0])
        fill_in.sort(key=lambda x: x[0])
        total_list = [0] * 1024
        i=0
        tally = 0
        pattern_count = 0
        while pattern_count < len(fill_in):
            if tally != fill_in[pattern_count][0]:
                total_list[i] = 1
                i+=1
                tally+=1
                continue
            if fill_in[pattern_count][2] == 0:
                total_list[i] = 0
                i+=1
                tally+=1
                pattern_count +=1
                continue

            if fill_in[pattern_count][2] == 1:
                    total_list[i+1] = fill_in[pattern_count][1] + 5
                    i+=1
            else:
                total_list[i] = fill_in[pattern_count][2]
                total_list[i+1] = fill_in[pattern_count][1] + 5
                i+=2
            
            tally += fill_in[pattern_count][2]
            pattern_count += 1

        total_list = total_list[:i]
        return total_list

    dataset = []
    for text_i in range(len(text_chunks)):
        sample_encodings, sample_indices = encode_text_with_indices(substitution_rule, symbols, text_chunks[text_i])
        sample_encodings = sample_encodings[:512]
        sample_indices = sample_indices[:512]
        if sample_indices.count(256) > 10:
            continue
        
        encodings_identifiers = unique_pattern_identifiers([sample_encodings], False)
        encodings_vocab = len(encodings_identifiers.items())

        encoding_list = invariate_sequence(sample_encodings, encodings_identifiers, encodings_vocab)
        dataset.append([encoding_list, sample_indices])
    return dataset



def filter_subset_pairs(pairs):

    def is_subset_pair(pair1, pair2):
        first1, second1 = set(pair1[0]), set(pair1[1])
        first2, second2 = set(pair2[0]), set(pair2[1])
        return (first1.issubset(first2) and second1.issubset(second2) and 
                (len(first1) < len(first2) or len(second1) < len(second2)))
    
    result = pairs.copy()
    i = len(result) - 1
    
    while i >= 0:
        should_remove = False
        for j, pair2 in enumerate(result):
            if i != j and is_subset_pair(result[i], pair2):
                should_remove = True
                break
        if should_remove:
            result.pop(i)
        i -= 1
        
    return result

def find_patterns_and_indices(sequence, remove_subsets=True):
    """
    Find all repeating subsequences in a sequence and their indices.
    Excludes indices of subsequences when they are part of a larger repeating subsequence.
    
    Args:
        sequence (list): Input sequence of numbers
        
    Returns:
        list: List of [subsequence, indices] pairs for repeating subsequences
    """
    n = len(sequence)
    result = []
    
    # Helper function to convert list to tuple for hashability
    def to_tuple(lst):
        return tuple(lst)
    
    # Find all possible subsequences and their indices
    subsequence_indices = {}
    for length in range(1, 5):  # Start from length 2
        for i in range(n - length + 1):
            subseq = to_tuple(sequence[i:i + length])
            if subseq not in subsequence_indices:
                subsequence_indices[subseq] = []
            subsequence_indices[subseq].append(i)
    
    # Filter out non-repeating subsequences
    repeating_subsequences = {
        subseq: indices 
        for subseq, indices in subsequence_indices.items() 
        if len(indices) > 1
    }
    
    # Sort subsequences by length (longest first)
    sorted_subsequences = sorted(
        repeating_subsequences.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    # Keep track of used indices
    used_indices = set()
    
    # Process subsequences from longest to shortest
    for subseq, indices in sorted_subsequences:
        # Filter out indices that are already part of longer subsequences
        valid_indices = []
        if remove_subsets:
            for idx in indices:
                # Check if any position in this occurrence overlaps with used indices
                overlap = False
                for pos in range(idx, idx + len(subseq)):
                    if pos in used_indices:
                        overlap = True
                        break
                if not overlap:
                    valid_indices.append(idx)
                    # Mark all positions in this occurrence as used
                    for pos in range(idx, idx + len(subseq)):
                        used_indices.add(pos)
        else:
            valid_indices = indices
            
        # Only add subsequence if it still has multiple valid occurrences
        if len(valid_indices) > 1:
            result.append([list(subseq), valid_indices])
    
    return result

def find_unique_singles(sequence):
    arr = []
    for i, element in enumerate(sequence):
        count = sequence.count(element)
        if count == 1:
            arr.append([[element], [i]])
    
    return arr
