import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import re
from typing import List, Tuple, Union, Dict
import gc

class OptimizedJobMatcher:
    """
    An optimized TF-IDF implementation for matching resumes to job descriptions
    at scale. Designed to efficiently handle large datasets (millions of documents)
    through batch processing and memory optimization.
    """
    
    def __init__(
        self, 
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        batch_size: int = 50000,
        custom_stop_words: List[str] = None
    ):
        """
        Initialize the job matcher with customizable parameters.
        
        Args:
            max_features: Maximum number of features to use in TF-IDF
            ngram_range: Range of n-grams to use (e.g., (1, 2) for unigrams and bigrams)
            batch_size: Number of documents to process at once
            custom_stop_words: Additional stop words specific to job descriptions
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.batch_size = batch_size
        
        # Prepare stop words
        self.custom_stop_words = [
            'experience', 'job', 'required', 'skill', 'work', 'team', 'position',
            'candidate', 'company', 'ability', 'looking', 'opportunity'
        ]
        if custom_stop_words:
            self.custom_stop_words.extend(custom_stop_words)
            
        # Initialize vectorizer
        self.vectorizer = None
        self.resume_vector = None
        
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces between words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fit_vectorizer(self, documents: List[str], resume_text: str) -> None:
        """
        Fit the TF-IDF vectorizer on a sample of documents and the resume.
        
        Args:
            documents: List of job description texts to sample from
            resume_text: The resume text to match against
        """
        # Preprocess the resume
        processed_resume = self._preprocess_text(resume_text)
        
        # Take a sample of documents for fitting the vectorizer
        sample_size = min(10000, len(documents))
        sample_indices = np.random.choice(len(documents), sample_size, replace=False)
        sample_docs = [self._preprocess_text(documents[i]) for i in sample_indices]
        
        # Add the resume to the sample
        sample_docs.append(processed_resume)
        
        # Initialize and fit the vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        
        # Add custom stop words
        self.vectorizer.fit(sample_docs)
        if hasattr(self.vectorizer, 'stop_words_'):
            self.vectorizer.stop_words_.update(self.custom_stop_words)
        
        # Transform just the resume
        self.resume_vector = self.vectorizer.transform([processed_resume])
        
        # Free memory
        del sample_docs
        gc.collect()
    
    def find_top_matches(
        self, 
        job_descriptions: List[str], 
        job_ids: List[Union[str, int]],
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Find the top matching job descriptions for the resume.
        
        Args:
            job_descriptions: List of job description texts
            job_ids: List of corresponding job IDs
            top_n: Number of top matches to return
            
        Returns:
            DataFrame with job_id and similarity score, sorted by similarity
        """
        if self.vectorizer is None or self.resume_vector is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        
        # Calculate number of batches
        n_docs = len(job_descriptions)
        n_batches = (n_docs + self.batch_size - 1) // self.batch_size
        
        # Store top matches
        top_matches = []
        min_score_to_beat = -1
        
        print(f"Processing {n_docs} documents in {n_batches} batches...")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, n_docs)
            
            # Get batch data
            batch_docs = job_descriptions[start_idx:end_idx]
            batch_ids = job_ids[start_idx:end_idx]
            
            # Preprocess batch documents
            processed_batch = [self._preprocess_text(doc) for doc in batch_docs]
            
            # Transform batch with vectorizer
            batch_vectors = self.vectorizer.transform(processed_batch)
            
            # Calculate similarities
            similarities = cosine_similarity(self.resume_vector, batch_vectors).flatten()
            
            # Process this batch's top matches
            batch_results = list(zip(batch_ids, similarities))
            
            # Only keep top matches
            if len(top_matches) < top_n:
                top_matches.extend(batch_results)
                top_matches.sort(key=lambda x: x[1], reverse=True)
                if len(top_matches) > top_n:
                    top_matches = top_matches[:top_n]
                    min_score_to_beat = top_matches[-1][1]
            else:
                # Only consider scores better than our current threshold
                for job_id, score in batch_results:
                    if score > min_score_to_beat:
                        top_matches.append((job_id, score))
                
                # Re-sort and trim
                top_matches.sort(key=lambda x: x[1], reverse=True)
                top_matches = top_matches[:top_n]
                min_score_to_beat = top_matches[-1][1]
            
            # Free memory
            del batch_vectors, processed_batch
            gc.collect()
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{n_batches}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(top_matches, columns=['job_id', 'similarity'])
        return results_df.sort_values('similarity', ascending=False)
    
    def extract_key_terms(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Extract the most important terms from the resume based on TF-IDF weights.
        
        Args:
            top_n: Number of top terms to extract
            
        Returns:
            List of (term, weight) tuples
        """
        if self.vectorizer is None or self.resume_vector is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get resume vector as array and find non-zero elements
        resume_array = self.resume_vector.toarray()[0]
        non_zero_indices = resume_array.nonzero()[0]
        
        # Create (term, weight) pairs and sort by weight
        term_weights = [(feature_names[i], resume_array[i]) for i in non_zero_indices]
        term_weights.sort(key=lambda x: x[1], reverse=True)
        
        return term_weights[:top_n]


# Example usage:
if __name__ == "__main__":
    # Sample data
    sample_jobs = [
        "Senior Python Developer with 5+ years experience in Django and Flask",
        "Data Scientist with expertise in machine learning and statistics",
        "Frontend Developer with React and JavaScript skills",
        # ... more job descriptions
    ]
    sample_ids = [1, 2, 3]
    
    resume = """
    Experienced Python Developer with 6 years working on web applications.
    Skilled in Django, Flask, and FastAPI. Strong background in database design
    and API development. Familiar with React for frontend work.
    """
    
    # Initialize and use the matcher
    matcher = OptimizedJobMatcher(max_features=1000)
    matcher.fit_vectorizer(sample_jobs, resume)
    
    # Find top matches
    top_matches = matcher.find_top_matches(sample_jobs, sample_ids, top_n=2)
    print("Top matching jobs:")
    print(top_matches)
    
    # Extract key terms from resume
    key_terms = matcher.extract_key_terms(top_n=10)
    print("\nKey terms in resume:")
    for term, weight in key_terms:
        print(f"{term}: {weight:.4f}")