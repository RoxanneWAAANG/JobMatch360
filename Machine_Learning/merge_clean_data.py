# merge_clean_data.py
import pandas as pd

def merge_and_clean_data(job_skills_path, job_summary_path, job_postings_path, output_path):
    # Load CSV files into dataframes
    job_skills_df = pd.read_csv(job_skills_path)
    job_summary_df = pd.read_csv(job_summary_path)
    job_postings_df = pd.read_csv(job_postings_path)

    # Merge job_postings with job_skills and job_summary on 'job_link'
    merged_df = job_postings_df.merge(job_skills_df, on='job_link', how='left')
    merged_df = merged_df.merge(job_summary_df, on='job_link', how='left')

    # Drop duplicate job postings and reset the index
    merged_df.drop_duplicates(subset='job_link', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Handle missing values: fill missing summaries or skills with an empty string
    merged_df['job_summary'] = merged_df['job_summary'].fillna('')
    merged_df['job_skills'] = merged_df['job_skills'].fillna('')

    # Create a unified text column for embeddings
    merged_df['job_text'] = merged_df['job_summary'] + ' ' + merged_df['job_skills']

    # Optionally, filter out rows with no content in 'job_text'
    merged_df = merged_df[merged_df['job_text'].str.strip() != '']

    # Save merged data to CSV
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

if __name__ == "__main__":
    # Update these paths as needed.
    job_skills_path = "dataset/job_skills.csv"
    job_summary_path = "dataset/job_summary.csv"
    job_postings_path = "dataset/linkedin_job_postings.csv"
    output_path = "dataset/merged_jobs.csv"
    
    merge_and_clean_data(job_skills_path, job_summary_path, job_postings_path, output_path)
