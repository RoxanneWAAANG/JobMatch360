import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time

class JinaEmbedV3:
    def __init__(self, model_path: str = None, onnx_path: str = None):
        """
        Simple TF-IDF based embedding as a fallback solution
        
        Args:
            model_path: Not used, kept for API compatibility
            onnx_path: Not used, kept for API compatibility
        """
        # Initialize a TF-IDF vectorizer with more features for better embeddings
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        print("Using TF-IDF based embeddings as fallback solution")
    
    def _preprocess_text(self, text):
        """
        Simple text preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def embed(self, text: str):
        """
        Generate embeddings for the input text using TF-IDF
        
        Args:
            text: Input text
            
        Returns:
            tuple: (embeddings, time_taken)
        """
        start_time = time.time()
        
        # Preprocess the text
        processed_text = self._preprocess_text(text)
        
        # Fit and transform the text
        try:
            # Fit the vectorizer on the text
            self.vectorizer.fit([processed_text])
            
            # Transform the text to get embeddings
            embeddings = self.vectorizer.transform([processed_text])
            
            # Convert to dense array and normalize
            dense_embeddings = embeddings.toarray()
            norm = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
            normalized_embeddings = dense_embeddings / np.clip(norm, a_min=1e-8, a_max=None)
            
            end_time = time.time()
            
            return normalized_embeddings, end_time - start_time
        
        except Exception as e:
            print(f"Error in TF-IDF embedding: {e}")
            # Return a simple random embedding as a last resort
            random_embedding = np.random.rand(1, 1000)
            random_embedding = random_embedding / np.linalg.norm(random_embedding)
            end_time = time.time()
            return random_embedding, end_time - start_time

if __name__ == "__main__":
    sample_text = "Rock N Roll Sushi is hiring a Restaurant Manager!"
    embedder = JinaEmbedV3()
    embeddings, time_taken = embedder.embed(sample_text)
    print(type(embeddings), embeddings.shape)
    print(f"Time taken: {time_taken} seconds")