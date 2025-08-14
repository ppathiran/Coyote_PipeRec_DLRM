import numpy as np

# Define the shape and dtype used when saving the array
num_entries = 4 * 1024 * 1024  # 4M
max_collisions = 9
array_shape = (num_entries, max_collisions)
dtype = np.uint64

def fnv1a_hash_64(word):
    FNV_prime = 0x1000000000000000000013B
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

print("I", hex(fnv1a_hash_64("I")))

hash_value = fnv1a_hash_64("I")
upper_42_bits = (hash_value >> 22) & 0x3FFFFFFFFFF
index_22_bits = hash_value & 0x3FFFFF


# Load the binary file into a numpy array
with open("hash_table.bin", "rb") as f:
    hash_table_array_loaded = np.fromfile(f, dtype=dtype).reshape(array_shape)

hex_values = [hex(value) for value in hash_table_array_loaded[index_22_bits]]
print(f"Index {hex(index_22_bits)}: {hex_values}")

# # Print the contents in hex format
# print("Loaded hash table array in hex format:")
# for row_idx, row in enumerate(hash_table_array_loaded[:100]):  # Adjust the range to print more or fewer rows
#     hex_values = [hex(value) for value in row]
#     print(f"Row {row_idx}: {hex_values}")

