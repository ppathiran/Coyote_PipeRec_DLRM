import numpy as np
from gensim.models import KeyedVectors

# Define the path for the pre-trained model
model_path = "/home/yuzhuyu/u55c/word2vec/GoogleNews-vectors-negative300.bin"

# Load the Word2Vec Model
print("Loading the Word2Vec model...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Get the vocabulary and embeddings
vocab = list(model.key_to_index.keys())
num_words = len(vocab)
embedding_dim = 300
padded_dim = 320  # 1280 bytes / 4 bytes per float = 320 floats

# Initialize a numpy array for the padded embeddings
padded_embeddings = np.zeros((num_words, padded_dim), dtype=np.float32)

# Fill in the embeddings, and pad with zeros
for index, word in enumerate(vocab):
    embedding = model[word]  # 300-dimensional embedding
    padded_embeddings[index, :embedding_dim] = embedding  # Place the 300 floats

# Save the array to a pure binary file
with open("padded_embeddings.bin", "wb") as f:
    padded_embeddings.tofile(f)

print("Padded embeddings saved to 'padded_embeddings.bin' as a pure binary file")
