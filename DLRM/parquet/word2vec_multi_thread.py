import gensim
from gensim.models import KeyedVectors
import nltk
import re
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel processing
import time

# Download NLTK tokenizer resources (if not downloaded already)
nltk.download('punkt')

model_path = "/home/yuzhuyu/u55c/word2vec/GoogleNews-vectors-negative300.bin"
# Directory path containing the text files you want to process
folder_path = "/home/yuzhuyu/u55c/word2vec/coca-samples-text/"

# Step 1: Preprocess and Tokenize the Text (without lowercasing)
def preprocess_text(text):
    # Remove special symbols like "@@" and "<p>" using regex
    text = re.sub(r'[@@<>]', '', text)

    # Tokenize the text using NLTK's word_tokenize (without lowercasing)
    tokens = nltk.word_tokenize(text)

    # Remove any non-alphabetic tokens (like punctuation)
    tokens = [word for word in tokens if word.isalpha()]
    
    return tokens

# Step 2: Load the Pre-Trained Word2Vec Model (load once)
def load_word2vec_model():
    print("Loading the Word2Vec model...")
    return KeyedVectors.load_word2vec_format(model_path, binary=True)

# Step 3: Read and Preprocess the Text File (in chunks)
def process_chunk(start_idx, end_idx, text):
    # Extract the chunk from the text
    chunk = text[start_idx:end_idx]
    return preprocess_text(chunk)

# Step 4: Convert Tokens to Embeddings (using the model)
def tokens_to_embeddings(tokens, model):
    embeddings = []
    for token in tqdm(tokens, desc="Processing tokens"):
        if token in model:
            embeddings.append(model[token])
        else:
            embeddings.append(np.random.randn(model.vector_size))  # OOV words handled with random vectors
    return embeddings

# Step 5: Split the file into chunks for parallel processing
def split_file_into_chunks(text, num_chunks):
    chunk_size = len(text) // num_chunks
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    return chunks

# Step 6: Process a single file
def process_single_file(file_path, model, num_threads):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the file into chunks for parallel processing
    chunks = split_file_into_chunks(text, num_threads)
    
    # Use joblib to parallelize the tokenization process for each chunk
    print(f"Processing file: {file_path}")
    all_tokens = Parallel(n_jobs=num_threads)(delayed(process_chunk)(start_idx, end_idx, text) for start_idx, end_idx in chunks)
    
    # Flatten the list of tokens from all chunks
    all_tokens_flat = [token for sublist in all_tokens for token in sublist]
    
    # Convert tokens to embeddings using the Word2Vec model
    embeddings = tokens_to_embeddings(all_tokens_flat, model)
    
    return embeddings

# Step 7: Process all files in the folder
def process_all_files_in_folder(folder_path, model, num_threads):
    all_embeddings = {}

    # Get a list of all text files in the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Use joblib to parallelize the processing of multiple files
    results = Parallel(n_jobs=num_threads)(delayed(process_single_file)(file_path, model, num_threads) for file_path in files)
    
    # Store the results
    for filename, embeddings in zip(files, results):
        all_embeddings[filename] = embeddings
        print(f"Finished processing {filename}, Embeddings shape: {embeddings.shape}")

    return all_embeddings

# Main function to process all files in the folder
def main():
    # Load the Word2Vec model once
    model = load_word2vec_model()

    # Define the number of threads you want to use (e.g., 4 threads)
    num_threads = 4  # Adjust this based on your system's capabilities
    
    # Call the function to process all text files in the folder
    start_time = time.time()
    all_embeddings = process_all_files_in_folder(folder_path, model, num_threads)
    end_time = time.time()

    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    return all_embeddings

# Run the main function
if __name__ == "__main__":
    all_embeddings = main()
