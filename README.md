# JobMatch360-Career-Recommendation-System

Demo: https://huggingface.co/spaces/reinashi/JobMatch360

Dataset: https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024?select=linkedin_job_postings.csv

Embedding Model: https://huggingface.co/jinaai/jina-embeddings-v3

# JobMatch360

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



