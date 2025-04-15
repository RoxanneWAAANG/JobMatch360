import pandas as pd

def filter_jobs(job_location=None, max_rows=100000):
    """
    Filter jobs by location and ensure they have descriptions.
    
    Args:
        job_location: Location to filter by (if None, no location filtering)
        max_rows: Maximum number of rows to return (random sample if exceeded)
        
    Returns:
        DataFrame with job_title and job_summary columns
    """
    # Load only necessary columns from linkedin_job_postings.csv
    print("Loading job postings metadata...")
    cols = ['job_link', 'job_title', 'job_location', 'got_summary', 'company']
    postings_df = pd.read_csv('data/linkedin_job_postings.csv', 
                             usecols=cols)
    
    # Filter for jobs with descriptions
    print("Filtering for jobs with descriptions...")
    postings_df = postings_df[postings_df['got_summary'] == 't']
    
    # Apply location filter if provided
    if job_location:
        print(f"Filtering for jobs in location: {job_location}")
        postings_df = postings_df[postings_df['job_location'].str.contains(job_location, case=False, na=False)]
    
    # If no jobs match the criteria, return an empty DataFrame
    if postings_df.empty:
        print("No jobs found matching the criteria")
        return pd.DataFrame(columns=['job_link', 'job_title', 'job_summary'])
    
    # Load job summaries for only the filtered job links
    print(f"Loading job summaries for {len(postings_df)} filtered jobs...")
    
    # Use a chunked approach to handle the large job_summary file
    chunks = []
    for chunk in pd.read_csv('data/job_summary.csv', chunksize=500000):
        # Filter the chunk to only include job links from our filtered dataset
        filtered_chunk = chunk[chunk['job_link'].isin(postings_df['job_link'])]
        chunks.append(filtered_chunk)
        
        # If we've found all our job links, we can stop
        if len(pd.concat(chunks)) >= len(postings_df):
            break
    
    # Combine chunks
    summaries_df = pd.concat(chunks, ignore_index=True)
    
    # Merge the dataframes
    result_df = pd.merge(postings_df, summaries_df, on='job_link')
    
    # Select only the columns we need
    result_df = result_df[['job_title', 'job_summary', 'job_link', 'company', 'job_location']]
    
    # Random sample if there are too many rows
    if len(result_df) > max_rows:
        print(f"Sampling {max_rows} rows from {len(result_df)} matches")
        result_df = result_df.sample(n=max_rows, random_state=42)
    
    print(f"Final dataset contains {len(result_df)} jobs")
    return result_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter LinkedIn jobs by location')
    parser.add_argument('--location', type=str, help='Location to filter (e.g., "New York")')
    parser.add_argument('--max_rows', type=int, default=100000, help='Maximum number of rows to return')
    parser.add_argument('--output', type=str, help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    filtered_jobs = filter_jobs(args.location, args.max_rows)
    
    if args.output:
        filtered_jobs.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    else:
        print(f"Found {len(filtered_jobs)} matching jobs")
        # Print a small sample
        print("\nSample of results:")
        print(filtered_jobs.head(3))
