import numpy as np
from collections import defaultdict
from gensim.models import KeyedVectors

# Define the path for the pre-trained model
model_path = "/home/yuzhuyu/u55c/word2vec/GoogleNews-vectors-negative300.bin"

# Load the Pre-Trained Word2Vec Model
print("Loading the Word2Vec model...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("Model structure: ", model)

# Get the vocabulary (list of words included in the model)
vocab = list(model.key_to_index.keys())

def fnv1a_hash_64(word):
    FNV_prime = 0x1000000000000000000013B # 315
    FNV_prime = 0x10000000000000000000201 # 513
    hash_val = 0x6c62272e07bb014262b821756295c58d
    for byte in word.encode():
        # print("  byte", hex(byte))
        hash_val ^= byte
        # print("    xor value", hex(hash_val))
        hash_val = (hash_val * FNV_prime) & 0xffffffffffffffffffffffffffffffff
        # print("      mul value", hex(hash_val))
        hash_quantized = hash_val & 0xFFFFFFFFFFFFFFFF
        # print("         quantized value", hex(hash_quantized))
    return hash_quantized

from collections import defaultdict

def xor_hash_64(word):
    hash_val = 0x6c62272e07bb0142  # Initial hash value, similar to an offset basis
    for byte in word.encode():
        hash_val ^= byte  # XOR with the byte
        hash_val = (hash_val << 5) | (hash_val >> (64 - 5))  # Left rotate by 5 bits
        hash_val &= 0xFFFFFFFFFFFFFFFF  # Keep it to 64 bits
    return hash_val

# Dictionary to count occurrences of each hash value
collision_counts = defaultdict(int)

# Populate the dictionary with hash counts from the vocabulary
for index, word in enumerate(vocab):
    full_hash = fnv1a_hash_64(word)
    full_hash_22 = full_hash & 0x3FFFFF  # Extract the lower 22 bits
    collision_counts[full_hash_22] += 1  # Count occurrences of each hash value

# Find the maximum collision count
max_collision = max(collision_counts.values())
print(f"Maximum number of collisions for a single hash value: {max_collision}")

# Optionally, find and print all hash values with this maximum collision count
most_collided_hashes = [hash_val for hash_val, count in collision_counts.items() if count == max_collision]
print(f"Hash values with the most collisions (count = {max_collision}):")
for hash_val in most_collided_hashes:
    print(hex(hash_val))



# # Track unique combined 64-bit values
# unique_combined_values = set()

# # Populate unique combined 64-bit values from vocabulary
# for index, word in enumerate(vocab):
#     full_hash = fnv1a_hash_64(word)
#     upper_42_bits = (full_hash >> 22) & 0x3FFFFFFFFFF
#     index_22_bits = index & 0x3FFFFF
#     combined_64_bit_value = (upper_42_bits << 22) | index_22_bits
#     unique_combined_values.add(combined_64_bit_value)

# for word in vocab[:10]:
#     print(word, hex(fnv1a_hash_64(word)))

# word = "I"
# full_hash = fnv1a_hash_64(word)
# upper_42_bits = (full_hash >> 22) & 0x3FFFFFFFFFF
# index_22_bits = vocab.index(word) & 0x3FFFFF  # Assuming the index in vocab corresponds to the transformation index
# transformed_hash = (upper_42_bits << 22) | index_22_bits


# if (transformed_hash & 0xFFFFFFFFFFFFFFFF) not in unique_combined_values:
#     print(transformed_hash, " does not exist in the list of combined 64-bit values.")

# if (0xFFFFFFFF & 0xFFFFFFFFFFFFFFFF) not in unique_combined_values:
#     print(0xFFFFFFFF, " does not exist in the list of combined 64-bit values.")

# if (0xFFFFFFFFFFFFFFFF & 0xFFFFFFFFFFFFFFFF) not in unique_combined_values:
#     print(0xFFFFFFFFFFFFFFFF, " does not exist in the list of combined 64-bit values.")


# Dictionary to store the hash tables
hash_table_dict = defaultdict(list)

# Populate the dictionary with 64-bit values
for index, word in enumerate(vocab):
    full_hash = xor_hash_64(word)
    upper_40_bits = (full_hash >> 24) & 0xFFFFFFFFFF
    lower_24_bits = full_hash & 0xFFFFFF
    index_24_bits = index & 0xFFFFFF
    combined_64_bit_value = (upper_40_bits << 24) | index_24_bits
    hash_table_dict[lower_24_bits].append(combined_64_bit_value)

# # Create a 2D numpy array with 4M rows and 9 columns, initialized to 0
# num_entries = 4 * 1024 * 1024  # 4M
# max_collisions = 20
# hash_table_array = np.zeros((num_entries, max_collisions), dtype=np.uint64)

# # Populate the array with a check for collisions
# for lower_22_bits, values in hash_table_dict.items():
#     if len(values) > max_collisions:
#         print(f"Warning: Collision count for index {lower_22_bits:06x} exceeds max_collisions ({len(values)} > {max_collisions})")
    
#     if (len(values) == 0):
#         print(f"Index {lower_22_bits:06x} has {len(values)} collisions")

    # if len(values) > 4:
    #     print(f"Index {lower_22_bits:06x} has {len(values)} collisions")
    
    # for slot, combined_64_bit_value in enumerate(values[:max_collisions]):
    #     hash_table_array[lower_22_bits, slot] = combined_64_bit_value



# # Populate the array
# for lower_22_bits, values in hash_table_dict.items():
#     for slot, combined_64_bit_value in enumerate(values[:max_collisions]):
#         hash_table_array[lower_22_bits, slot] = combined_64_bit_value

# # Save the array to a pure binary file
# with open("hash_table.bin", "wb") as f:
#     hash_table_array.tofile(f)

# print("Hash table saved to 'hash_table.bin' as a pure binary file")

# print("I", hex(fnv1a_hash_64("I")))


# print("think", hex(fnv1a_hash_64("think")))
# print("it", hex(fnv1a_hash_64("it")))
# print("is", hex(fnv1a_hash_64("is")))
# print("safe", hex(fnv1a_hash_64("safe")))
