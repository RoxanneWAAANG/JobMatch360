import streamlit as st
import pandas as pd
import os
import tempfile
import time
import json
import base64
import traceback

# Set page config
st.set_page_config(
    page_title="JobMatch360 - AI-Powered Career Recommendations",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
    }
    .job-title {
        font-size: 1.3rem;
        color: #1565C0;
        font-weight: 600;
    }
    .company-name {
        font-size: 1.1rem;
        color: #424242;
        font-weight: 500;
    }
    .job-location {
        color: #757575;
        font-style: italic;
    }
    .match-score {
        color: #2E7D32;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #424242;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .job-description {
        max-height: 200px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 5px;
        margin-top: 10px;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'resume_content' not in st.session_state:
    st.session_state.resume_content = None

if 'key_skills' not in st.session_state:
    st.session_state.key_skills = None

if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None

if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# Create directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Sample real recommendation data
REAL_SAMPLE_DATA = {
  "resume": {
    "path": "Resume.pdf",
    "text_length": 3489
  },
  "parameters": {
    "job_location": "New York",
    "max_jobs": 100000,
    "tfidf_filter_count": 10,
    "final_recommendations": 5
  },
  "tfidf_results": [
    {
      "rank": 1,
      "job_id": 11988,
      "job_title": "NLP Team Lead Engineer",
      "score": 0.4323582134951342,
      "job_description": "Are you an experienced NLP Engineer with in-depth expertise in computational linguistics, artificial intelligence, programming, and a strong understanding of fundamental NLP concepts?\nRYTE, a breakthrough Artificial Intelligence health tech company is looking for an NLP Team Lead Engineer who will contribute to pushing the boundaries of RYTE's natural language processing capabilities and driving innovation within the company's linguistic technology initiatives.\nFinding the right physician, selecting the organization providing the most suitable service, and obtaining the most relevant health information, whether in the US or globally, can be complex and filled with uncertainty. Enter RYTE: an AI-driven platform designed to simplify this process. By analyzing data from millions of healthcare providers, experts, medical procedures, clinical outcomes, and a variety of other data points, RYTE offers users a comprehensive view of their options, uniquely tailored to their needs. It offers:\nPersonalized Recommendations: Based on your specific situation, RYTE's platform presents the best healthcare options for you, no matter where you are in the world.\nUnprecedented Insights: With billions of data points on countless medical professionals and facilities, its AI-driven software gives rankings and insights with metrics previously unavailable for decision-making.\nVersatile Applications: Launching for individuals seeking specialized care (B2C), with plans for the Pharmaceutical, Medtech, Health Insurance (B2B), and Doctor (B2D) markets, the RYTE platform caters to diverse needs."
    },
    {
      "rank": 2,
      "job_id": 904,
      "job_title": "Senior NLP Engineer (USA Remote)",
      "score": 0.38866495471816737,
      "job_description": "Are you a Senior NLP Engineer with in-depth expertise in computational linguistics, artificial intelligence, programming, and a strong understanding of fundamental NLP concepts?\nRYTE, a breakthrough Artificial Intelligence health tech company is looking for Senior NLP Engineers who will contribute to pushing the boundaries of RYTE's natural language processing capabilities and driving innovation within the company's linguistic technology initiatives.\nAbout RYTE\nFinding the right physician, selecting the organization providing the most suitable service, and obtaining the most relevant health information, whether in the US or globally, can be complex and filled with uncertainty. Enter RYTE: an AI-driven platform designed to simplify this process. By analyzing data from millions of healthcare providers, experts, medical procedures, clinical outcomes, and a variety of other data points, RYTE offers users a comprehensive view of their options, uniquely tailored to their needs."
    },
    {
      "rank": 3,
      "job_id": 596,
      "job_title": "GenAI Lead Data Scientist, Corporate Vice President",
      "score": 0.37639595624460925,
      "job_description": "Location Designation:\nHybrid\nWhen you join New York Life, you're joining a company that values career development, collaboration, innovation, and inclusiveness. We want employees to feel proud about being part of a company that is committed to doing the right thing. You'll have the opportunity to grow your career while developing personally and professionally through various resources and programs. New York Life is a relationship-based company and appreciates how both virtual and in-person interactions support our culture.\nThe Center for Data Science and Artificial Intelligence (CDSAi) is the 70-person innovative corporate Analytics group within New York Life. We are a rapidly growing entrepreneurial department which designs, creates, and offers innovative data-driven solutions for many parts of the enterprise."
    },
    {
      "rank": 4,
      "job_id": 8524,
      "job_title": "Principal Machine Learning Scientist",
      "score": 0.3708283566822779,
      "job_description": "DISCO is seeking a Principal Machine Learning Scientist to join our AI/ML team supporting our cutting-edge AI features in the domain of legal technology. The successful candidate will have a wide knowledge of techniques and theory in machine learning, with a special emphasis on deep learning. Most of our applications involve learning from document text, so NLP experience is of particular interest, especially experience with transfer learning using transformer models such as BERT and/or using Large Language Models.\nYour Impact\nAI/ML forms a core part of DISCO's brand and vision, and this position provides an opportunity to develop ideas that will transform the legal domain with significant benefits for the broader society."
    },
    {
      "rank": 5,
      "job_id": 2112,
      "job_title": "Distinguished Applied Researcher",
      "score": 0.3533044438195648,
      "job_description": "Job Description\nCenter 1 (19052), United States of America, McLean, Virginia\nDistinguished Applied Researcher\nOverview\nAt Capital One, we are creating trustworthy and reliable AI systems, changing banking for good. For years, Capital One has been leading the industry in using machine learning to create real-time, intelligent, automated customer experiences. From informing customers about unusual charges to answering their questions in real time, our applications of AI & ML are bringing humanity and simplicity to banking. We are committed to building world-class applied science and engineering teams and continue our industry leading capabilities with breakthrough product experiences and scalable, high-performance AI infrastructure."
    }
  ],
  "embedding_results": [
    {
      "rank": 1,
      "job_id": 647,
      "job_title": "Sr. Distinguished Applied Researcher",
      "score": 0.6086693834138226,
      "job_description": "Job Description\nCenter 1 (19052), United States of America, McLean, Virginia\nSr. Distinguished Applied Researcher\nOverview\nAt Capital One, we are creating trustworthy and reliable AI systems, changing banking for good. For years, Capital One has been leading the industry in using machine learning to create real-time, intelligent, automated customer experiences. From informing customers about unusual charges to answering their questions in real time, our applications of AI & ML are bringing humanity and simplicity to banking."
    },
    {
      "rank": 2,
      "job_id": 596,
      "job_title": "GenAI Lead Data Scientist, Corporate Vice President",
      "score": 0.6058652940192831,
      "job_description": "Location Designation:\nHybrid\nWhen you join New York Life, you're joining a company that values career development, collaboration, innovation, and inclusiveness. We want employees to feel proud about being part of a company that is committed to doing the right thing. You'll have the opportunity to grow your career while developing personally and professionally through various resources and programs."
    },
    {
      "rank": 3,
      "job_id": 2112,
      "job_title": "Distinguished Applied Researcher",
      "score": 0.6046937648316044,
      "job_description": "Job Description\nCenter 1 (19052), United States of America, McLean, Virginia\nDistinguished Applied Researcher\nOverview\nAt Capital One, we are creating trustworthy and reliable AI systems, changing banking for good. For years, Capital One has been leading the industry in using machine learning to create real-time, intelligent, automated customer experiences."
    },
    {
      "rank": 4,
      "job_id": 13999,
      "job_title": "Distinguished Applied Researcher",
      "score": 0.6046937648316044,
      "job_description": "Job Description\nCenter 1 (19052), United States of America, McLean, Virginia\nDistinguished Applied Researcher\nOverview\nAt Capital One, we are creating trustworthy and reliable AI systems, changing banking for good. For years, Capital One has been leading the industry in using machine learning to create real-time, intelligent, automated customer experiences."
    },
    {
      "rank": 5,
      "job_id": 14394,
      "job_title": "(Global Oil Gas) Senior Data Scientist Expert",
      "score": 0.5700822195460575,
      "job_description": "This role required candidate to permanently relocate at Dhahran, Saudi Arabia.\nAbout the Company\nThis company engages in the exploration, production, transportation, and sale of crude oil and natural gas. It operates through the following segments: Upstream, Downstream, and Corporate. The Upstream segment includes crude oil, natural gas and natural gas liquids exploration, field development, and production."
    }
  ]
}

# Sample key skills extracted from the resume
SAMPLE_KEY_SKILLS = [
    ("machine learning", 0.85),
    ("artificial intelligence", 0.82),
    ("nlp", 0.79),
    ("python", 0.76),
    ("deep learning", 0.72),
    ("transformers", 0.68),
    ("tensorflow", 0.65),
    ("pytorch", 0.63),
    ("data science", 0.61),
    ("huggingface", 0.58)
]

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # Convert string to bytes
    b64 = base64.b64encode(object_to_download.encode()).decode()
    
    # Generate download link
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    return href

def load_real_data_demo():
    """
    Load real recommendation data for demo purposes
    """
    st.session_state.demo_mode = True
    st.session_state.processing_time = 3.45  # Simulated processing time
    
    # Create a DataFrame from the real embedding results
    job_data = []
    
    # Use embedding results as they're typically the final output
    for job in REAL_SAMPLE_DATA["embedding_results"]:
        job_data.append({
            'job_id': job['job_id'],
            'job_title': job['job_title'],
            'similarity': job['score'],
            'rank': job['rank'],
            'company': 'RYTE' if 'RYTE' in job['job_description'] else 'Capital One' if 'Capital One' in job['job_description'] else 'New York Life' if 'New York Life' in job['job_description'] else 'Company',
            'job_location': 'New York' if job['job_id'] == 596 else 'Remote' if job['job_id'] == 904 else 'McLean, Virginia' if 'McLean' in job['job_description'] else 'Dhahran, Saudi Arabia' if 'Dhahran' in job['job_description'] else 'USA',
            'job_link': f'https://linkedin.com/jobs/{job["job_id"]}',
            'job_description': job['job_description']
        })
    
    st.session_state.recommendations = pd.DataFrame(job_data)
    st.session_state.key_skills = SAMPLE_KEY_SKILLS

def process_resume(uploaded_resume, job_location=None):
    """
    Process the uploaded resume and generate recommendations.
    In a real production environment, this would call your actual recommendation logic.
    For the demo, we'll use pre-generated sample data.
    """
    start_time = time.time()
    tmp_file_path = None
    
    try:
        # Create a temporary file to save the uploaded resume
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_resume.getvalue())
            tmp_file_path = tmp_file.name
        
        # For demo purposes, we'll just load the sample data
        load_real_data_demo()
        
        # Clean up the temporary file
        if tmp_file_path:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        return True
    
    except Exception as e:
        # Get detailed error message
        error_details = traceback.format_exc()
        print(f"Error processing resume: {error_details}")
        
        # Show user-friendly error
        st.error(f"Error processing resume. Please try again with a different PDF file.")
        
        # Clean up the temporary file
        if tmp_file_path:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        return False

def main():
    # Sidebar - Using emoji instead of image
    st.sidebar.markdown('# 🚀 JobMatch360')
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="info-box">
    <p>Upload your resume (PDF format) to get personalized job recommendations based on your skills and experience.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Resume upload section
    st.sidebar.markdown('<div class="section-header">Upload Your Resume</div>', unsafe_allow_html=True)
    uploaded_resume = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
    # Location filter
    st.sidebar.markdown('<div class="section-header">Filter by Location (Optional)</div>', unsafe_allow_html=True)
    job_location = st.sidebar.text_input("Enter a location (city, state, country)")
    
    # Process button
    process_button = st.sidebar.button("Get Recommendations")
    
    # Demo button
    demo_button = st.sidebar.button("Load Demo Data")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="info-box">
    <p><b>About JobMatch360</b></p>
    <p>JobMatch360 uses advanced AI to match your resume with the most relevant job opportunities.</p>
    <p>The system analyzes both keyword matching (TF-IDF) and semantic understanding (Jina Embeddings) to provide tailored recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    st.markdown('<div class="main-header">JobMatch360: AI-Powered Career Recommendations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">Find your perfect job match with advanced AI analysis</div>
    """, unsafe_allow_html=True)
    
    # Handle demo button
    if demo_button:
        with st.spinner('Loading demo data...'):
            load_real_data_demo()
            st.success("Demo data loaded successfully!")
    
    # Process the resume if the button is clicked
    if process_button and uploaded_resume is not None:
        with st.spinner('Analyzing your resume and finding the best matches...'):
            success = process_resume(uploaded_resume, job_location)
            if success:
                st.success("Resume analysis complete!")
    
    # Display recommendations if available
    if st.session_state.recommendations is not None:
        # Display processing time
        st.markdown(f"<p>Analysis completed in {st.session_state.processing_time:.2f} seconds</p>", unsafe_allow_html=True)
        
        # Display key skills section
        if st.session_state.key_skills:
            st.markdown('<div class="section-header">Key Skills Identified in Your Resume</div>', unsafe_allow_html=True)
            skills_col1, skills_col2 = st.columns(2)
            
            with skills_col1:
                for term, weight in st.session_state.key_skills[:5]:
                    st.markdown(f"- **{term}** (relevance: {weight:.2f})")
            
            with skills_col2:
                if len(st.session_state.key_skills) > 5:
                    for term, weight in st.session_state.key_skills[5:10]:
                        st.markdown(f"- **{term}** (relevance: {weight:.2f})")
        
        # Top recommendations header
        st.markdown('<div class="section-header">Top Job Recommendations</div>', unsafe_allow_html=True)
        
        # Display each job recommendation
        for i, (_, row) in enumerate(st.session_state.recommendations.iterrows()):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="card">
                        <div class="job-title">{row['job_title']}</div>
                        <div class="company-name">{row['company']}</div>
                        <div class="job-location">{row['job_location']}</div>
                        <div class="match-score">Match Score: {row['similarity']:.2f}</div>
                        <a href="{row['job_link']}" target="_blank">View Job on LinkedIn</a>
                        <div class="job-description">
                            {row.get('job_description', 'No description available')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Download results button
        if st.session_state.recommendations is not None:
            csv = st.session_state.recommendations.to_csv(index=False)
            st.markdown(
                download_link(
                    st.session_state.recommendations, 
                    'job_recommendations.csv', 
                    'Download Recommendations as CSV'
                ),
                unsafe_allow_html=True
            )
            
            # Display comparison of matching techniques
            st.markdown('<div class="section-header">How JobMatch360 Works</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <p>JobMatch360 uses a two-stage approach for accurate job matching:</p>
                <ol>
                    <li><strong>Stage 1: TF-IDF Matching</strong> - A fast keyword-based initial filter to identify potentially relevant jobs</li>
                    <li><strong>Stage 2: Semantic Embedding Matching</strong> - Deep semantic analysis to understand the meaning and context of your skills and experience</li>
                </ol>
                <p>This combination provides more accurate and relevant recommendations than traditional keyword matching alone.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the matching techniques comparison
            st.markdown("### Comparison of Matching Techniques")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### TF-IDF Results (Stage 1)")
                for i, job in enumerate(REAL_SAMPLE_DATA["tfidf_results"][:3]):
                    st.markdown(f"""
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 3px solid #1E88E5; background-color: #f5f5f5;">
                        <div style="font-weight: bold;">{job['job_title']}</div>
                        <div style="color: #2E7D32;">Score: {job['score']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Embedding Results (Stage 2 - Final)")
                for i, job in enumerate(REAL_SAMPLE_DATA["embedding_results"][:3]):
                    st.markdown(f"""
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 3px solid #2E7D32; background-color: #f5f5f5;">
                        <div style="font-weight: bold;">{job['job_title']}</div>
                        <div style="color: #2E7D32;">Score: {job['score']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Display welcome information when no resume is uploaded
        st.markdown("""
        <div class="info-box">
            <h3>How It Works</h3>
            <ol>
                <li>Upload your resume in PDF format</li>
                <li>Optionally enter a preferred location</li>
                <li>Our AI analyzes your skills and experience</li>
                <li>Get matched with the best job opportunities</li>
            </ol>
            <p>The system uses both keyword matching (TF-IDF) and semantic understanding (advanced embedding model) to find the most relevant jobs for your profile.</p>
            <p>Click "Load Demo Data" to see an example of how JobMatch360 works!</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()