import torch
import gensim
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

# Download necessary resources for NLTK tokenization
nltk.download('punkt')

# Function to load a pre-trained Word2Vec model (Gensim format)
def load_word2vec_model(model_path):
    """
    Load pre-trained Word2Vec model.
    :param model_path: Path to the Word2Vec model.
    :return: Loaded model.
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

# Function to preprocess text (lowercase and tokenize)
def preprocess_text(text):
    """
    Preprocess the input text: lowercase, tokenize.
    :param text: Raw text input.
    :return: List of tokens.
    """
    tokens = word_tokenize(text.lower())
    return tokens

# Function to convert tokens into embeddings using a pre-trained Word2Vec model
def tokens_to_embeddings(tokens, model):
    """
    Convert tokens to embeddings using the Word2Vec model.
    :param tokens: List of tokens.
    :param model: Pre-trained Word2Vec model.
    :return: Tensor of embeddings.
    """
    embeddings = []
    for token in tokens:
        if token in model:
            embeddings.append(model[token])
        else:
            # Optional: Handle out-of-vocabulary words (OOV) with a random vector
            embeddings.append(np.random.randn(model.vector_size))
    
    # Convert list of embeddings to a PyTorch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings_tensor

# Function to load and process the text file, and generate embeddings
def text_file_to_embeddings(file_path, model_path):
    """
    Read a text file, preprocess it, and convert it to embeddings.
    :param file_path: Path to the UTF-8 encoded text file.
    :param model_path: Path to the pre-trained Word2Vec model.
    :return: Tensor of embeddings.
    """
    # Step 1: Load the Word2Vec model
    word2vec_model = load_word2vec_model(model_path)
    
    # Step 2: Read and preprocess the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    tokens = preprocess_text(text)
    
    # Step 3: Convert tokens to embeddings
    embeddings = tokens_to_embeddings(tokens, word2vec_model)
    
    return embeddings

# Example usage
if __name__ == "__main__":
    # Path to the text file and pre-trained Word2Vec model
    text_file_path = "/home/yuzhuyu/u55c/word2vec/coca-samples-text/text_acad.txt"
    word2vec_model_path = "/home/yuzhuyu/u55c/word2vec/word2vec_model.bin"  # Pre-trained Word2Vec (binary format)

    # Generate embeddings
    embeddings_tensor = text_file_to_embeddings(text_file_path, word2vec_model_path)

    # Print the embeddings tensor shape
    print(f"Embeddings shape: {embeddings_tensor.shape}")
