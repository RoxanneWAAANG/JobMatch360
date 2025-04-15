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

To achieve good recommendations, we compared naive, machine learning, and deep learning approaches. The Linkedin data set contains 1.3M jobs, too handle such large amount of data, we conbined machine learning with deep learning, see secitons below for more details. 

## Table of Contents

- [Naive Approach](#naive-approach)
- [Installation](#installation)
- [Machine Learning Approach](#machine-learning-approach)
- [Deep Learning Approach](#deep-learning-approach)
- [Result Comparison](#result-comparison)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Ethics Statement](#ethics-statement)
- [License](#license)

## Installation

1. Clone the Repository

```bash
git clone https://github.com/RoxanneWAAANG/JobMatch360.git
cd JobMatch360
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Naive Approach

This Naive Approach will randomly select 5 jobs from the dataset, comparing the job skills + summary to our sample resume text via cosine similarity.

To run, run 
```bash
python naive.py
```

## Machine Learning Approach
Machine learning approach generate job embeddings using TF-IDF

## Deep Learning Approach

Deep learning approach uses advanced NLP techniques to match a job seeker's resume to relevant job postings. It employs a two-stage recommendation process:

1. **TF-IDF Filtering**: First, it uses TF-IDF (Term Frequency-Inverse Document Frequency) to quickly filter a large dataset of job postings down to a manageable subset of potential matches.

2. **Semantic Embedding Matching**: Then, it uses the Jina Embeddings v3 model to create semantic embeddings of both the resume and filtered job descriptions, finding the most semantically similar jobs through cosine similarity.

--- 

1. Install dependencies

2. Download the dataset:
   - Download CSV files from [LinkedIn Jobs and Skills Dataset on Kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024/)
   - Place the downloaded files in the `dl/data/` directory

3. Download the Jina Embeddings v3 model:
```bash
wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx -O dl/jina-embeddings-v3/onnx/model.onnx
wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx_data -O dl/jina-embeddings-v3/onnx/model.onnx_data
wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model_fp16.onnx -O dl/jina-embeddings-v3/onnx/model_fp16.onnx
```

4. Place your resume PDF in the root directory of the project, then run the recommendation system:
```bash
cd Deep_Learning
python recommend.py
```

## Result Comparison

Metric here is cosine similarity:

| Metric           | Naive Approach | ML Approach | DL Approach |
|------------------|----------------|-------------|-------------|
| Top 1 Similarity | 0.151          | 0.487       | 0.531       |
| Top 2 Similarity | 0.146          | 0.460       | 0.529       |
| Top 3 Similarity | 0.127          | 0.456       | 0.529       |
| Top 4 Similarity | 0.071          | 0.439       | 0.475       |
| Top 5 Similarity | 0.062          | 0.437       | 0.472       |

## Project Structure

```
.
├── LICENSE
├── README.md
├── Machine_Learning
│   ├── content_based_recommender.py
│   ├── generate_embeddings.py
│   └── merge_clean_data.py
├──Naive
│   └── naive.py
├── app
│   ├── app.py
│   ├── embedding.py
│   ├── filter_jobs.py
│   ├── recommend.py
│   ├── resume_parser.py
│   └── tfidf_match.py
├── dl
│   ├── embedding.py
│   ├── filter_jobs.py
│   ├── recommend.py
│   ├── resume_parser.py
│   ├── tfidf_match.py
│   └── jina-embeddings-v3/onnx/
└── dataset
    └── merged_common_subset_100.csv        # 100 sample of the dataset
```

## Requirements

- Tested on Python 3.11
- Dependencies listed in `requirements.txt`:

## License

This project is licensed under the terms found in the LICENSE file.

## Ethics Statement

This job recommendation system is built with the following ethical principles:

- **Privacy Protection**: All resume data is anonymized, removing personal identifiers including emails, phone numbers, and addresses.

- **Transparency**: The system logs provide insight into how recommendations are generated, helping users understand match rationale.

- **User Control**: Users maintain ownership of their data and can adjust recommendation parameters to suit their preferences.

- **Accessibility**: We strive for an inclusive system that serves diverse users across different devices and platforms.

- **Compliance**: The design respects relevant data protection regulations while following AI ethics best practices.

This system aims to augment traditional job search methods, not replace human judgment in the hiring process.

## Acknowledgments

- [LinkedIn Jobs and Skills Dataset](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024/) for job data
- [Jina AI](https://huggingface.co/jinaai/jina-embeddings-v3) for the embeddings model
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF processing
