import gensim
from gensim.models import KeyedVectors
import nltk
import re
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for the progress bar
import torch  # Import PyTorch
import time

# Download NLTK tokenizer resources (if not downloaded already)
nltk.download('punkt')

# Step 1: Preprocess and Tokenize the Text (without lowercasing)
def preprocess_text(text):
    # Remove special symbols like "@@" and "<p>" using regex
    text = re.sub(r'[@@<>]', '', text)

    # Tokenize the text using NLTK's word_tokenize (without lowercasing)
    tokens = nltk.word_tokenize(text)

    # Remove any non-alphabetic tokens (like punctuation)
    tokens = [word for word in tokens if word.isalpha()]
    
    return tokens

# Step 3: Read and Preprocess the Text File
def process_text_file(file_path):
    # Read the content of the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Preprocess and tokenize the text
    tokens = preprocess_text(text)
    
    return tokens

# Step 4: Convert Word2Vec model to PyTorch tensor and move to GPU
def convert_word2vec_to_pytorch(model, device):
    word_vectors = model.vectors  # Get the Word2Vec vectors
    return torch.tensor(word_vectors, device=device), model.index_to_key  # Return vectors and corresponding tokens

# Step 5: Convert Tokens to Embeddings using PyTorch
def tokens_to_embeddings(tokens, word2vec_tensors, word2vec_vocab, device):
    word2vec_tensor, vocab = word2vec_tensors
    embeddings = []
    
    vocab_index = {word: idx for idx, word in enumerate(vocab)}  # Create a vocab index
    
    # for token in tqdm(tokens, desc="Processing tokens"):
    for token in tokens:
        if token in vocab_index:
            embedding = word2vec_tensor[vocab_index[token]]  # Lookup embedding
            embeddings.append(embedding)
        else:
            # Handle out-of-vocabulary (OOV) words by using a random vector or ignoring them
            random_vector = torch.randn(word2vec_tensor.size(1), device=device)  # Generate random vector on the GPU
            embeddings.append(random_vector)
    
    # Convert list of embeddings to a tensor
    return torch.stack(embeddings)

# Step 6: Process all files in the folder and get embeddings
def process_all_files_in_folder(folder_path, word2vec_tensors, word2vec_vocab, device):
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        start_time = time.time()
        # Only process .txt files
        if filename.endswith(".txt"):
            print(f"\nProcessing file: {filename}")
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")

            # Tokenize and get embeddings
            start_time_tokenize = time.time()
            tokens = process_text_file(file_path)
            end_time_tokenize = time.time()
            print(f"Time taken for tokenizing {filename}: {end_time_tokenize - start_time_tokenize:.2f} seconds")

            start_time_embeddings = time.time()
            embeddings = tokens_to_embeddings(tokens, word2vec_tensors, word2vec_vocab, device)
            end_time_embeddings = time.time()
            print(f"Time taken for embeddings {filename}: {end_time_embeddings - start_time_embeddings:.2f} seconds")
            
            # Check the shape of the embeddings for this file
            print(f"Embeddings shape for {filename}: {embeddings.shape}")

            # Discard the embeddings after processing and free up GPU memory
            del embeddings  # Remove embeddings from memory
            torch.cuda.empty_cache()  # Free up GPU memory

            end_time = time.time()
            print(f"Time taken for processing {filename}: {end_time - start_time:.2f} seconds")


# Define the path for the pre-trained model
model_path = "/home/yuzhuyu/u55c/word2vec/GoogleNews-vectors-negative300.bin"
# model_path = "/home/yuzhuyu/word2vec/GoogleNews-vectors-negative300.bin"

folder_path = "/home/yuzhuyu/u55c/word2vec/coca-samples-text/"
# folder_path = "/home/yuzhuyu/word2vec/coca-samples-text/"

# Step 2: Load the Pre-Trained Word2Vec Model
print("Loading the Word2Vec model...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("Word2Vec model loaded.")

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert Word2Vec to PyTorch tensors and move to GPU
word2vec_tensors = convert_word2vec_to_pytorch(model, device)
word2vec_vocab = model.index_to_key

# Call the function to process all text files in the folder

for i in range(3):
    process_all_files_in_folder(folder_path, word2vec_tensors, word2vec_vocab, device)
 

# Example of accessing embeddings for a specific file:
# embeddings_for_file = all_embeddings['text_acad.txt']
