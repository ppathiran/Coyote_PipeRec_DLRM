import gensim
from gensim.models import KeyedVectors
import nltk
import re
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for the progress bar
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

# Step 4: Convert Tokens to Embeddings (Batch Processing)
def tokens_to_embeddings_batch(tokens, model, batch_size=512):
    """
    Converts tokens to embeddings using batch processing.
    Args:
    - tokens: List of tokens.
    - model: Pre-trained Word2Vec model.
    - batch_size: Number of tokens to process in a single batch.

    Returns:
    - embeddings: Numpy array of shape (len(tokens), vector_size).
    """
    embeddings = []
    for i in tqdm(range(0, len(tokens), batch_size), desc="Processing tokens in batches"):
        batch_tokens = tokens[i:i + batch_size]

        # Separate tokens into in-vocabulary (IV) and out-of-vocabulary (OOV)
        iv_tokens = [token for token in batch_tokens if token in model]
        oov_count = len(batch_tokens) - len(iv_tokens)

        # Retrieve embeddings for in-vocabulary tokens
        if iv_tokens:
            iv_embeddings = model[iv_tokens]
            embeddings.extend(iv_embeddings)
        
        # Handle out-of-vocabulary tokens (random vectors)
        if oov_count > 0:
            oov_embeddings = np.random.randn(oov_count, model.vector_size)
            embeddings.extend(oov_embeddings)

    return np.array(embeddings)

# Step 5: Process all files in the folder and get embeddings
def process_all_files_in_folder(folder_path, model, batch_size=512):
    all_embeddings = {}
    
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        start_time = time.time()
        # Only process .txt files
        if filename.endswith(".txt"):
            print(f"\nProcessing file: {filename}")
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")

            start_time_tokenize = time.time()
            tokens = process_text_file(file_path)
            end_time_tokenize = time.time()
            print(f"Time taken for tokenizing {filename}: {end_time_tokenize - start_time_tokenize:.2f} seconds")

            start_time_embedding = time.time()
            embeddings = tokens_to_embeddings_batch(tokens, model, batch_size=batch_size)
            end_time_embedding = time.time()
            print(f"Time taken for converting tokens to embeddings for {filename}: {end_time_embedding - start_time_embedding:.2f} seconds")
            
            # Store embeddings with the filename as the key
            all_embeddings[filename] = embeddings
            
            # Check the shape of the embeddings for this file
            print(f"Embeddings shape for {filename}: {embeddings.shape}")

        end_time = time.time()
        print(f"Time taken for processing {filename}: {end_time - start_time:.2f} seconds")
    
    return all_embeddings

# Define the path for the pre-trained model
model_path = "/home/yuzhuyu/u55c/word2vec/GoogleNews-vectors-negative300.bin"
# Directory path containing the text files you want to process
folder_path = "/home/yuzhuyu/u55c/word2vec/coca-samples-text/"

# Call the function to process all text files in the folder
time_start = time.time()
# Step 2: Load the Pre-Trained Word2Vec Model
print("Loading the Word2Vec model...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
time_load = time.time()
print(f"Time taken to load the Word2Vec model: {time_load - time_start:.2f} seconds")

# Process all files in the folder and get embeddings
# Repeat 3 times to get the average time
for i in range(3):
    all_embeddings = process_all_files_in_folder(folder_path, model, batch_size=512)

# You now have embeddings for all text files in the 'all_embeddings' dictionary
# Example of accessing embeddings for a specific file:
# embeddings_for_file = all_embeddings['text_acad.txt']
