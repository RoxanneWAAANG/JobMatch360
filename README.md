# JobMatch360-Career-Recommendation-System

Demo: https://huggingface.co/spaces/reinashi/JobMatch360

Dataset: https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024?select=linkedin_job_postings.csv

Embedding Model: https://huggingface.co/jinaai/jina-embeddings-v3

# JobMatch360

Despite the fact that current job boards such as LinkedIn and Glassdoor provide recommendations based off of the users, it is not an ideal system
Users will still have to view the exact job requirements, and are even encouraged to tailor their resume to make sure they can show themselves as an ideal candidate (when most often than not the main goal is to hit those keywords to not get auto flagged by the system)
We think this is the reverse of how it should be.

Our approach to a recommendation system considers the combination of job skills and job summaries and seeks to match it more exact towards your resume
We are not tailoring our resumes to match jobs, we are tailoring the jobs to match the resume.

JobMatch360 is a LinkedIn-style career recommendation system that leverages machine learning techniques to match users with relevant job opportunities based on their resume or profile. The project demonstrates how to merge and clean data from multiple sources, generate job embeddings using TF-IDF, and build a basic content-based recommender system.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [License](#license)

## Overview

JobMatch360 processes job posting data from three sources:
- **job_skills_sub.csv**: Contains job links and associated skills.
- **job_summary_sub.csv**: Contains job links and job summaries.
- **linkedin_job_postings_sub.csv**: Contains detailed job posting metadata.

The system:
1. Merges these datasets based on the `job_link` field.
2. Extracts common job entries across all datasets.
3. Generates job embeddings using a TF-IDF vectorizer.
4. Provides job recommendations using cosine similarity between a user's resume and job postings.

## Features

- **Data Merging & Cleaning:**  
  Combines data from multiple CSV files while handling missing values and ensuring consistency.

- **Subset Extraction:**  
  Creates a 100-row subset containing only the common job instances across all sources.

- **Job Embeddings:**  
  Uses TF-IDF to convert job descriptions and skills into numerical vectors.

- **Content-Based Recommendation:**  
  Recommends jobs to users by calculating cosine similarity between user profiles and job postings.

## Project Structure

```
.
├── LICENSE
├── README.md
├── checkpoints
│   ├── job_embeddings.pkl
│   └── tfidf_vectorizer.pkl
├── content_based_recommender.py
├── dataset
│   ├── create_subset.ipynb
│   ├── job_skills.csv
│   ├── job_summary.csv
│   ├── linkedin_job_postings.csv
│   └── merged_common_subset_100.csv
├── generate_embeddings.py
└── merge_clean_data.py
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/RoxanneWAAANG/JobMatch360.git
   cd JobMatch360
   ```
## Naive Approach

This Naive Approach will randomly select 5 jobs from the dataset, comparing the job skills + summary to our sample resume text via cosine similarity.

To run, run 
```bash
python naive.py
```

## Machine Learning Approach

### Step 1: Merge and Clean Data

Run the `merge_clean_data.py` script to merge and clean the datasets:

```bash
python merge_clean_data.py
```

This script creates a file named `merged_data.csv`.

### Step 2: Generate Job Embeddings

Generate TF-IDF embeddings from the merged dataset by running:

```bash
python generate_embeddings.py
```

This creates two files:
- `checkpoints/job_embeddings.pkl` – The saved TF-IDF embeddings.
- `checkpoints/tfidf_vectorizer.pkl` – The saved TF-IDF vectorizer model.

### Step 3: Run the Content-Based Recommender

Test the recommender system with a sample resume by running:

```bash
python content_based_recommender.py
```

The script loads the merged data and the saved embeddings, then outputs the top job recommendations based on the provided resume text.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



