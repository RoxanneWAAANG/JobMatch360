import numpy as np
import pandas as pd
from tfidf_match import OptimizedJobMatcher
from embedding import JinaEmbedV3
from resume_parser import ResumeParser
from typing import List, Dict, Tuple, Union
import time

class JobRecommender:
    def __init__(
        self, 
        embedding_model_path: str = 'jinaai/jina-embeddings-v3',
        embedding_onnx_path: str = 'jina-embeddings-v3/onnx/model.onnx',
        tfidf_max_features: int = 5000,
        tfidf_batch_size: int = 50000
    ):
        # Initialize embedding model
        self.embedder = JinaEmbedV3(model_path=embedding_model_path, onnx_path=embedding_onnx_path)
        
        # Initialize TF-IDF matcher
        self.tfidf_matcher = OptimizedJobMatcher(
            max_features=tfidf_max_features,
            batch_size=tfidf_batch_size
        )
        
        # Initialize resume parser
        self.resume_parser = ResumeParser(anonymize=True)
    
    def _extract_resume_text(self, resume_pdf_path: str) -> str:
        """Extract full text from resume PDF"""
        parsed_resume = self.resume_parser.parse(resume_pdf_path)
        return parsed_resume.get("FULL_CONTENT", "")
    
    def recommend(
        self, 
        resume_pdf_path: str,
        job_location: str = None,
        max_jobs: int = 100000,
        tfidf_filter_count: int = 50,
        final_recommendations: int = 5
    ) -> pd.DataFrame:
        """
        Generate job recommendations in two stages:
        1. Use TF-IDF to filter to a smaller set (e.g., top 50)
        2. Use embeddings for precise semantic matching on the filtered set
        
        Args:
            resume_pdf_path: Path to the resume PDF
            job_location: Location to filter jobs by (e.g., "New York")
            max_jobs: Maximum number of jobs to process
            tfidf_filter_count: Number of jobs to filter using TF-IDF
            final_recommendations: Number of final recommendations to return
            
        Returns:
            DataFrame with job_id, similarity score, and rank
        """
        start_time = time.time()
        
        # Setup logging
        import logging
        import os
        import json
        from datetime import datetime
        
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_dir}/job_recommendation_{timestamp}.log"
        details_filename = f"{log_dir}/job_recommendation_details_{timestamp}.json"
        
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger()
        
        logger.info(f"Starting job recommendation process")
        logger.info(f"Resume: {resume_pdf_path}, Location: {job_location}")
        
        # Get filtered jobs dataset
        from filter_jobs import filter_jobs
        jobs_df = filter_jobs(job_location=job_location, max_rows=max_jobs)
        
        # Extract job descriptions and create job IDs
        job_descriptions = jobs_df['job_summary'].tolist()
        job_titles = jobs_df['job_title'].tolist()
        job_ids = list(range(len(job_descriptions)))
        
        print(f"Starting recommendation process for {len(job_descriptions)} jobs")
        logger.info(f"Processing {len(job_descriptions)} jobs")
        
        # Extract resume text
        resume_text = self._extract_resume_text(resume_pdf_path)
        print(f"Extracted resume text ({len(resume_text)} chars)")
        logger.info(f"Extracted resume text ({len(resume_text)} chars)")
        
        # Stage 1: TF-IDF filtering
        print(f"Stage 1: TF-IDF filtering to top {tfidf_filter_count} matches")
        logger.info(f"Stage 1: TF-IDF filtering to top {tfidf_filter_count} matches")
        
        self.tfidf_matcher.fit_vectorizer(job_descriptions, resume_text)
        tfidf_matches = self.tfidf_matcher.find_top_matches(
            job_descriptions=job_descriptions,
            job_ids=job_ids,
            top_n=tfidf_filter_count
        )
        
        # Log TF-IDF results
        logger.info("TF-IDF Top Matches:")
        tfidf_results = []
        for i, (_, row) in enumerate(tfidf_matches.iterrows()):
            job_id = int(row['job_id'])  # Convert numpy.float64 to int
            score = row['similarity']
            title = job_titles[job_id]
            logger.info(f"  {i+1}. {title} (ID: {job_id}, Score: {score:.4f})")
            tfidf_results.append({
                "rank": i+1,
                "job_id": job_id,
                "job_title": title,
                "score": float(score),
                "job_description": job_descriptions[job_id]
            })
        
        # Get filtered job IDs and descriptions
        filtered_job_ids = [int(job_id) for job_id in tfidf_matches['job_id'].tolist()]  # Convert to int
        filtered_job_desc_map = {
            job_id: job_descriptions[i] 
            for i, job_id in enumerate(job_ids) 
            if job_id in filtered_job_ids
        }
        filtered_job_descriptions = [filtered_job_desc_map[job_id] for job_id in filtered_job_ids]
        
        print(f"Stage 1 complete: Filtered to {len(filtered_job_ids)} jobs")
        logger.info(f"Stage 1 complete: Filtered to {len(filtered_job_ids)} jobs")
        
        # Stage 2: Embedding-based matching
        print("Stage 2: Computing semantic embeddings")
        logger.info("Stage 2: Computing semantic embeddings")
        
        # Embed the resume
        resume_embedding, resume_time = self.embedder.embed(resume_text)
        print(f"Resume embedding computed in {resume_time:.2f} seconds")
        logger.info(f"Resume embedding computed in {resume_time:.2f} seconds")
        
        # Embed each filtered job description and compute similarities
        similarities = []
        for i, (job_id, job_text) in enumerate(zip(filtered_job_ids, filtered_job_descriptions)):
            job_embedding, _ = self.embedder.embed(job_text)
            # Compute cosine similarity
            similarity = np.dot(resume_embedding, job_embedding.T)[0][0]
            similarities.append((job_id, similarity, job_titles[job_id], job_text))
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(filtered_job_ids)} embeddings")
                logger.info(f"Processed {i + 1}/{len(filtered_job_ids)} embeddings")
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:final_recommendations]
        
        # Log embedding results
        logger.info("Embedding Top Matches:")
        embedding_results = []
        for i, (job_id, similarity, title, job_text) in enumerate(top_matches):
            logger.info(f"  {i+1}. {title} (ID: {job_id}, Score: {similarity:.4f})")
            embedding_results.append({
                "rank": i+1,
                "job_id": job_id,
                "job_title": title,
                "score": float(similarity),
                "job_description": job_text
            })
        
        # Create results DataFrame (without job description for return value)
        results = pd.DataFrame([(j[0], j[1], j[2]) for j in top_matches], 
                              columns=['job_id', 'similarity', 'job_title'])
        results['rank'] = range(1, len(results) + 1)
        
        # Save detailed results to JSON file
        detailed_results = {
            "resume": {
                "path": resume_pdf_path,
                "text_length": len(resume_text)
            },
            "parameters": {
                "job_location": job_location,
                "max_jobs": max_jobs,
                "tfidf_filter_count": tfidf_filter_count,
                "final_recommendations": final_recommendations
            },
            "tfidf_results": tfidf_results,
            "embedding_results": embedding_results
        }
        
        with open(details_filename, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"Recommendation process completed in {total_time:.2f} seconds")
        logger.info(f"Recommendation process completed in {total_time:.2f} seconds")
        logger.info(f"Log saved to: {log_filename}")
        logger.info(f"Detailed results saved to: {details_filename}")
        print(f"Detailed results saved to: {details_filename}")
        
        return results

if __name__ == "__main__":
    recommender = JobRecommender()
    recommendations = recommender.recommend(
        resume_pdf_path="JinglongXiong_resume_ml.pdf", # change this to your resume path
        job_location="New York", # change this to your job location
        max_jobs=100000,
        tfidf_filter_count=10,
        final_recommendations=5
    )
    
    print("\nTop Recommendations:")
    print(recommendations)
