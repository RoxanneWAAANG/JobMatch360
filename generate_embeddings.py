import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_job_embeddings(merged_data_path, embeddings_output_path, vectorizer_output_path):
    # Load merged data
    df = pd.read_csv(merged_data_path)
    
    # Ensure job_text column exists; if not, create it using job_summary and job_skills
    if 'job_text' not in df.columns:
        df['job_text'] = df['job_summary'].fillna('') + ' ' + df['job_skills'].fillna('')
    else:
        # Fill missing values in job_text with empty string
        df['job_text'] = df['job_text'].fillna('')
    
    # Create a TF-IDF Vectorizer and transform the job_text column
    tfidf = TfidfVectorizer(stop_words='english')
    job_embeddings = tfidf.fit_transform(df['job_text'])
    
    # Save the embeddings and the vectorizer to disk
    with open(embeddings_output_path, 'wb') as f:
        pickle.dump(job_embeddings, f)
    with open(vectorizer_output_path, 'wb') as f:
        pickle.dump(tfidf, f)
    
    print(f"Job embeddings saved to {embeddings_output_path}")
    print(f"TF-IDF vectorizer saved to {vectorizer_output_path}")

if __name__ == "__main__":
    merged_data_path = "dataset/merged_common_subset_100.csv"
    embeddings_output_path = "checkpoints/job_embeddings.pkl"
    vectorizer_output_path = "checkpoints/tfidf_vectorizer.pkl"
    
    generate_job_embeddings(merged_data_path, embeddings_output_path, vectorizer_output_path)
