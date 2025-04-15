import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def naive(user_text, df):

    #Convert the user text into a vectorizer
    tfidf = TfidfVectorizer()
    
    #Subset 5 random rows. These will be our "recommendations"
    sub_df = df.sample(n = 5, random_state = 0).copy()

    #Ensure job_text column exists
    if 'job_text' not in sub_df.columns:
        sub_df['job_text'] = sub_df['job_summary'].fillna('') + ' ' + sub_df['job_skills'].fillna('')
    else:
        sub_df['job_text'] = sub_df['job_text'].fillna('')

    #Combining both to prevent overfitting
    corpus = sub_df['job_text'].tolist() + [user_text]
    corpus_vectors = tfidf.fit_transform(corpus)

    user_vector = corpus_vectors[-1]
    sub_vectors = corpus_vectors[:-1]

    #Calculate the similarity scores
    sim_scores = cosine_similarity(user_vector, sub_vectors).flatten()

    #Create a new column for similarity scores
    sub_df['sim_scores'] = sim_scores

    #Sort the dataframe by the similarity scores
    top_recs = sub_df.sort_values(by="sim_scores", ascending=False)

    return top_recs[['job_link', 'job_title', 'company', 'job_location', 'sim_scores']]

if __name__ == "__main__":

    #Load the data (sample data in this case)
    df = pd.read_csv('dataset/merged_common_subset_100.csv')

    #Example sample resume
    sample_resume = """
    Experienced hospitality manager with expertise in customer service, restaurant management,
    team supervision, and operations. Skilled in training, scheduling, and inventory management.
    """

    #Get random 5 recommendations
    recommendations = naive(sample_resume, df)
    print("Recommendations based on the sample resume:")
    print(recommendations)