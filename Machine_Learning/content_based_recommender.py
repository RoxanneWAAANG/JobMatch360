# content_based_recommender.py
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

def load_embeddings_and_vectorizer(embeddings_path, vectorizer_path):
    with open(embeddings_path, 'rb') as f:
        job_embeddings = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        tfidf = pickle.load(f)
    return job_embeddings, tfidf

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()  # Extracts text from each page
    return text

def recommend_jobs(user_resume, df, job_embeddings, tfidf, top_n=5):
    """
    Given a user's resume/profile text, compute cosine similarity between
    the user's profile and job postings and return the top N recommendations.
    """
    # Transform the resume text into the TF-IDF space
    user_embedding = tfidf.transform([user_resume])
    
    # Calculate cosine similarity scores between the user profile and all job postings
    similarity_scores = cosine_similarity(user_embedding, job_embeddings).flatten()
    
    # Get the indices of the top N similar job postings
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # Extract the top recommendations and add similarity scores
    recommended_jobs = df.iloc[top_indices].copy()
    recommended_jobs['similarity_score'] = similarity_scores[top_indices]
    
    return recommended_jobs[['job_link', 'job_title', 'company', 'job_location', 'similarity_score']]

if __name__ == "__main__":
    # Load the merged job data
    merged_data_path = "dataset/merged_jobs.csv"
    df = pd.read_csv(merged_data_path)
    
    # Load the saved embeddings and TF-IDF vectorizer
    embeddings_path = "checkpoints/job_embeddings.pkl"
    vectorizer_path = "checkpoints/tfidf_vectorizer.pkl"
    job_embeddings, tfidf = load_embeddings_and_vectorizer(embeddings_path, vectorizer_path)
    
    # Example PDF resume path
    pdf_resume_path = "Roxanne_s_Resume__MLE_.pdf"
    
    # Extract the text from the PDF resume
    extracted_resume = extract_text_from_pdf(pdf_resume_path)
    
    # Get top 5 job recommendations based on the extracted resume text
    recommended = recommend_jobs(extracted_resume, df, job_embeddings, tfidf, top_n=5)
    
    print("Top job recommendations based on the PDF resume:")
    print(recommended)
