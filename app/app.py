import streamlit as st
import pandas as pd
import base64
import time
import tempfile

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
        margin-bottom: 2rem;
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
        margin-bottom: 1.5rem;
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
    .highlight {
        background-color: #FFF9C4;
        padding: 2px 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None

if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# Top 5 job recommendations with real data from JSON
RECOMMENDATIONS = [
    {
        "rank": 1,
        "job_id": 647,
        "job_title": "Sr. Distinguished Applied Researcher",
        "score": 0.6086693834138226,
        "job_description": "Job Description\nCenter 1 (19052), United States of America, McLean, Virginia\nSr. Distinguished Applied Researcher\nOverview\nAt Capital One, we are creating trustworthy and reliable AI systems, changing banking for good. For years, Capital One has been leading the industry in using machine learning to create real-time, intelligent, automated customer experiences. From informing customers about unusual charges to answering their questions in real time, our applications of AI & ML are bringing humanity and simplicity to banking. We are committed to building world-class applied science and engineering teams and continue our industry leading capabilities with breakthrough product experiences and scalable, high-performance AI infrastructure.",
        "company": "Capital One",
        "job_location": "McLean, Virginia",
        "job_link": "https://www.linkedin.com/jobs/view/sr-distinguished-applied-researcher-at-capital-one-3838647/"
    },
    {
        "rank": 2,
        "job_id": 596,
        "job_title": "GenAI Lead Data Scientist, Corporate Vice President",
        "score": 0.6058652940192831,
        "job_description": "Location Designation:\nHybrid\nWhen you join New York Life, you're joining a company that values career development, collaboration, innovation, and inclusiveness. We want employees to feel proud about being part of a company that is committed to doing the right thing. You'll have the opportunity to grow your career while developing personally and professionally through various resources and programs. New York Life is a relationship-based company and appreciates how both virtual and in-person interactions support our culture.\nThe Center for Data Science and Artificial Intelligence (CDSAi) is the 70-person innovative corporate Analytics group within New York Life. We are a rapidly growing entrepreneurial department which designs, creates, and offers innovative data-driven solutions for many parts of the enterprise.",
        "company": "New York Life",
        "job_location": "New York",
        "job_link": "https://www.linkedin.com/jobs/view/genai-lead-data-scientist-corporate-vice-president-at-new-york-life-3793596/"
    },
    {
        "rank": 3,
        "job_id": 2112,
        "job_title": "Distinguished Applied Researcher",
        "score": 0.6046937648316044,
        "job_description": "Job Description\nCenter 1 (19052), United States of America, McLean, Virginia\nDistinguished Applied Researcher\nOverview\nAt Capital One, we are creating trustworthy and reliable AI systems, changing banking for good. For years, Capital One has been leading the industry in using machine learning to create real-time, intelligent, automated customer experiences. From informing customers about unusual charges to answering their questions in real time, our applications of AI & ML are bringing humanity and simplicity to banking. We are committed to building world-class applied science and engineering teams and continue our industry leading capabilities with breakthrough product experiences and scalable, high-performance AI infrastructure.",
        "company": "Capital One",
        "job_location": "McLean, Virginia",
        "job_link": "https://www.linkedin.com/jobs/view/distinguished-applied-researcher-at-capital-one-3852112/"
    },
    {
        "rank": 4,
        "job_id": 13999,
        "job_title": "Distinguished Applied Researcher",
        "score": 0.6046937648316044,
        "job_description": "Job Description\nCenter 1 (19052), United States of America, McLean, Virginia\nDistinguished Applied Researcher\nOverview\nAt Capital One, we are creating trustworthy and reliable AI systems, changing banking for good. For years, Capital One has been leading the industry in using machine learning to create real-time, intelligent, automated customer experiences. From informing customers about unusual charges to answering their questions in real time, our applications of AI & ML are bringing humanity and simplicity to banking. We are committed to building world-class applied science and engineering teams and continue our industry leading capabilities with breakthrough product experiences and scalable, high-performance AI infrastructure.",
        "company": "Capital One",
        "job_location": "McLean, Virginia",
        "job_link": "https://www.linkedin.com/jobs/view/distinguished-applied-researcher-at-capital-one-3913999/"
    },
    {
        "rank": 5,
        "job_id": 14394,
        "job_title": "(Global Oil Gas) Senior Data Scientist Expert",
        "score": 0.5700822195460575,
        "job_description": "This role required candidate to permanently relocate at Dhahran, Saudi Arabia.\nAbout the Company\nThis company engages in the exploration, production, transportation, and sale of crude oil and natural gas. It operates through the following segments: Upstream, Downstream, and Corporate. The Upstream segment includes crude oil, natural gas and natural gas liquids exploration, field development, and production. The Downstream segment focuses on refining, logistics, power generation, and the marketing of crude oil, petroleum and petrochemical products, and related services to international and domestic customers.",
        "company": "Saudi Aramco",
        "job_location": "Dhahran, Saudi Arabia",
        "job_link": "https://www.linkedin.com/jobs/view/senior-data-scientist-expert-at-saudi-aramco-3914394/"
    }
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

def load_demo_data():
    """
    Load the demo data for the pitch
    """
    st.session_state.demo_mode = True
    st.session_state.processing_time = 3.45  # Processing time for the demo
    
    # Create a DataFrame from the recommendations
    job_data = []
    for job in RECOMMENDATIONS:
        job_data.append({
            'job_id': job['job_id'],
            'job_title': job['job_title'],
            'similarity': job['score'],
            'rank': job['rank'],
            'company': job['company'],
            'job_location': job['job_location'],
            'job_link': job['job_link'],
            'job_description': job['job_description']
        })
    
    st.session_state.recommendations = pd.DataFrame(job_data)

def process_resume(uploaded_resume, job_location=None):
    """
    Process the uploaded resume and generate recommendations.
    This is a simulated function that would call the actual recommendation model in production.
    """
    start_time = time.time()
    tmp_file_path = None
    
    try:
        # Create a temporary file to save the uploaded resume
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_resume.getvalue())
            tmp_file_path = tmp_file.name
        
        # Simulate processing time to make the demo feel realistic
        time.sleep(2.5)
        
        # Load demo data (in a real system, this would be the actual model output)
        load_demo_data()
        
        end_time = time.time()
        st.session_state.processing_time = end_time - start_time
        
        return True
    
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")
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
    
    # Get Recommendations button
    process_button = st.sidebar.button("Get Recommendations")
    
    # Demo button
    demo_button = st.sidebar.button("Load Demo Data")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="info-box">
    <p><b>About JobMatch360</b></p>
    <p>JobMatch360 uses advanced AI to match your resume with the most relevant job opportunities.</p>
    <p>The system analyzes both keyword matching (TF-IDF) and semantic understanding (advanced embedding models) to provide tailored recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    st.markdown('<div class="main-header">JobMatch360: AI-Powered Career Recommendations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">Find your perfect job match with advanced AI analysis</div>
    """, unsafe_allow_html=True)
    
    # Handle demo button if clicked
    if demo_button:
        with st.spinner('Loading demo data...'):
            load_demo_data()
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
        
        # Top recommendations header
        st.markdown('<div class="section-header">Top Job Recommendations</div>', unsafe_allow_html=True)
        
        # Display each job recommendation
        for i, (_, row) in enumerate(st.session_state.recommendations.iterrows()):
            with st.container():
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
        csv = st.session_state.recommendations.to_csv(index=False)
        st.markdown(
            download_link(
                st.session_state.recommendations, 
                'job_recommendations.csv', 
                'Download Recommendations as CSV'
            ),
            unsafe_allow_html=True
        )
        
        # How it works section
        st.markdown("""
        <div class="info-box">
            <p><b>How JobMatch360 Works</b></p>
            <p>JobMatch360 uses a two-stage approach for accurate job matching:</p>
            <ol>
                <li><strong>Stage 1: TF-IDF Matching</strong> - Fast keyword-based initial filtering</li>
                <li><strong>Stage 2: Semantic Embedding Matching</strong> - Deep understanding of skills and experience</li>
            </ol>
            <p>This AI-powered approach finds opportunities that keyword matching alone would miss.</p>
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
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()