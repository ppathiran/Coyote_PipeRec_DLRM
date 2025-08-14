import gensim
from gensim.models import KeyedVectors

# Define the path for the pre-trained model
model_path = "/home/yuzhuyu/u55c/word2vec/GoogleNews-vectors-negative300.bin"

# Load the Pre-Trained Word2Vec Model
print("Loading the Word2Vec model...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("Model structure: ", model)

# Get the vocabulary (list of words included in the model)
vocab = list(model.key_to_index.keys())

# Print the number of words in the vocabulary
print(f"Total number of words in the model's vocabulary: {len(vocab)}")

# Example: print the first 20 words in the vocabulary
print("First 100 words in the vocabulary:")
print(vocab[:100])

vectors = model.vectors
# print("First 20 word vectors:")
# print(vectors[:20])
print("Shape of the vectors array:", vectors.shape)
print("Size of each word vector:", model.vector_size)


max_len = 0

for word, index in list(model.key_to_index.items()):#[:20]:
    if (len(word) > max_len):
        max_len = len(word)
        print(f"Word: {word}, Index: {index}, size: {len(word)}")
    # print(f"Word: {word}, size: {len(word)}")
    # print(f"{hash(word) & 0xFFFFFFFFFFFFFFFF:X}, {hash(word)}")

print(f"Max word length: {max_len}")


# Calculate the total length of all words with the specified conditions
total_length = sum(len(word) if len(word) > 25 else 25 for word in vocab)
# Calculate the average length
average_length = total_length / len(vocab)
print(f"Average word length in the vocabulary: {average_length:.2f}")
print(f"Total length of all words in the vocabulary: {total_length}")
print(f"Number of words: ", len(vocab))



# # Optionally, you can write the vocabulary to a file if you want to review it in detail
# with open('GoogleNews-vectors-vocab.txt', 'w') as f:
#     for word in vocab:
#         f.write(word + '\n')

# print("Vocabulary has been written to 'GoogleNews-vectors-vocab.txt'")
