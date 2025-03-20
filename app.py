import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎨 **Enhanced CSS with Animations**
st.markdown("""
    <style>
    /* Reset default Streamlit styles for better control */
    body {
        background: #0A0F1F;
        color: #F5F8FF;
        font-family: 'Montserrat', sans-serif;
        margin: 0;
        padding: 0;
    }
    .stApp {
        background: linear-gradient(135deg, #1C253B, #0A0F1F);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0, 255, 255, 0.2);
        min-height: 100vh;
        position: relative;
        z-index: 1;
        animation: fadeIn 1s ease-in-out;
    }
    h1 {
        text-align: center;
        color: #00D4FF;
        font-size: 56px;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.8);
        animation: glow 1.5s ease-in-out infinite alternate, slideInFromTop 1s ease-out;
        margin-bottom: 40px;
    }
    h2 {
        color: #5CE1E6;
        font-size: 32px;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(92, 225, 230, 0.5);
        transition: transform 0.3s ease, color 0.3s ease;
        animation: bounceIn 0.8s ease-out;
    }
    h2:hover {
        transform: scale(1.05);
        color: #FF6F61;
    }
    .stTextArea, .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid #5CE1E6;
        border-radius: 15px;
        padding: 20px;
        color: #F5F8FF;
        box-shadow: 0 5px 20px rgba(92, 225, 230, 0.3);
        transition: all 0.4s ease;
        animation: fadeInUp 0.8s ease-out;
    }
    .stTextArea:hover, .stFileUploader:hover {
        border-color: #00D4FF;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.6);
        transform: translateY(-5px);
    }
    .stButton>button {
        background: linear-gradient(135deg, #00D4FF, #5CE1E6);
        color: #0A0F1F;
        padding: 15px 30px;
        font-size: 20px;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.5);
        transition: all 0.3s ease;
        animation: pulse 2s infinite ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5CE1E6, #FF6F61);
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(92, 225, 230, 0.7);
    }
    .stDataFrame {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.3);
        transition: transform 0.3s ease;
        animation: slideInFromLeft 1s ease-out;
    }
    .stDataFrame:hover {
        transform: translateY(-5px);
    }
    .stProgress .st-bo {
        background: #5CE1E6;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(92, 225, 230, 0.6);
        animation: progressPulse 1.5s infinite ease-in-out;
    }
    /* Success Modal Styling */
    .success-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #00D4FF, #5CE1E6);
        color: #0A0F1F;
        padding: 30px;
        border-radius: 15px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.8);
        animation: popIn 0.5s ease-out, glowModal 2s infinite ease-in-out;
        z-index: 1000;
        width: 80%;
        max-width: 500px;
    }
    .success-modal::before {
        content: '🎉 ✨ 🎉';
        position: absolute;
        top: -20px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 40px;
        animation: confettiFall 2s infinite;
    }
    .success-modal::after {
        content: '✨ 🎉 ✨';
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 40px;
        animation: confettiRise 2s infinite;
    }
    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    @keyframes slideInFromTop {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    @keyframes slideInFromLeft {
        0% { transform: translateX(-50px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    @keyframes fadeInUp {
        0% { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    @keyframes bounceIn {
        0% { transform: scale(0.5); opacity: 0; }
        60% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); }
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 5px 15px rgba(0, 212, 255, 0.5); }
        50% { transform: scale(1.05); box-shadow: 0 8px 25px rgba(0, 212, 255, 0.8); }
        100% { transform: scale(1); box-shadow: 0 5px 15px rgba(0, 212, 255, 0.5); }
    }
    @keyframes progressPulse {
        0% { box-shadow: 0 0 10px rgba(92, 225, 230, 0.6); }
        50% { box-shadow: 0 0 20px rgba(92, 225, 230, 1); }
        100% { box-shadow: 0 0 10px rgba(92, 225, 230, 0.6); }
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px #00D4FF, 0 0 20px #00D4FF; }
        to { text-shadow: 0 0 20px #00D4FF, 0 0 30px #5CE1E6; }
    }
    @keyframes glowModal {
        0% { box-shadow: 0 10px 30px rgba(0, 212, 255, 0.8); }
        50% { box-shadow: 0 15px 50px rgba(0, 212, 255, 1); }
        100% { box-shadow: 0 10px 30px rgba(0, 212, 255, 0.8); }
    }
    @keyframes popIn {
        0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
        80% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
        100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    }
    @keyframes confettiFall {
        0% { transform: translateX(-50%) translateY(-100%); opacity: 1; }
        100% { transform: translateX(-50%) translateY(100%); opacity: 0; }
    }
    @keyframes confettiRise {
        0% { transform: translateX(-50%) translateY(100%); opacity: 1; }
        100% { transform: translateX(-50%) translateY(-100%); opacity: 0; }
    }
    /* Ensure visibility of Streamlit elements */
    .stMarkdown, .stInfo, .stError {
        color: #F5F8FF !important;
        z-index: 2;
        animation: fadeIn 1s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# 🚀 **Page Title**
st.title("🌟 AI Resume Ranking System")

# 🎨 **Interactive Layout**
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Drop your PDF resumes here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple resumes in PDF format."
    )
    if uploaded_files:
        st.info(f"📑 {len(uploaded_files)} resume(s) uploaded successfully!")

with col2:
    st.header("📝 Job Description")
    job_description = st.text_area(
        "Paste the job description here...",
        height=150,
        placeholder="Enter details about the role..."
    )
    if job_description:
        st.info("✅ Job description added!")

# 📌 **Extract Text from PDF Resumes**
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = "".join([page.extract_text() or "" for page in pdf.pages])
    return text.strip()

# 📌 **Rank Resumes Using TF-IDF & Cosine Similarity**
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_desc_vector], resume_vectors).flatten()

# 📌 **AI Suggestions with Emojis**
def generate_resume_tips(score):
    if score > 80:
        return "🌟 Amazing match! Your resume shines bright!"
    elif score > 60:
        return "👍 Solid match! Add more keywords for a boost."
    else:
        return "🚀 Needs work! Highlight skills and key terms."

# 📌 **Interactive Ranking Process**
if st.button("🚀 Rank Resumes Now", use_container_width=True):
    if uploaded_files and job_description:
        st.header("📊 Resume Rankings")

        resumes = [extract_text_from_pdf(file) for file in uploaded_files]

        # Animated Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
            status_text.text(f"Processing... {i+1}%")
        status_text.text("✅ Analysis Complete!")

        # Calculate Scores
        scores = rank_resumes(job_description, resumes)

        # Results DataFrame
        results_df = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Match Score (%)": (scores * 100).round(2),
            "AI Tips": [generate_resume_tips(score * 100) for score in scores]
        }).sort_values(by="Match Score (%)", ascending=False)

        # Styling with Gradient
        def highlight_score(val):
            intensity = min(1, val / 100)
            color = f'background: linear-gradient(90deg, rgba(92, 225, 230, {intensity}), rgba(255, 255, 255, 0.1));'
            return f'{color} color: white; font-weight: bold; text-align: center;'

        styled_df = results_df.style.applymap(highlight_score, subset=["Match Score (%)"]).set_properties(**{
            'border-radius': '10px',
            'padding': '10px',
        })

        # Display Results
        st.dataframe(styled_df, use_container_width=True)

        # Beautiful Success Modal
        st.markdown(
            f"""
            <div class="success-modal">
                🎉 Top Match: <strong>{results_df.iloc[0]['Resume']}</strong> with {results_df.iloc[0]['Match Score (%)']}%!
            </div>
            """,
            unsafe_allow_html=True
        )

        # Download Button
        csv = progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
            status_text.text(f"Processing... {i+1}%")
        status_text.text("✅ Analysis Complete!")

        # Calculate Scores
        scores = rank_resumes(job_description, resumes)

        # Results DataFrame
        results_df = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Match Score (%)": (scores * 100).round(2),
            "AI Tips": [generate_resume_tips(score * 100) for score in scores]
        }).sort_values(by="Match Score (%)", ascending=False)

        # Styling with Gradient
        def highlight_score(val):
            intensity = min(1, val / 100)
            color = f'background: linear-gradient(90deg, rgba(92, 225, 230, {intensity}), rgba(255, 255, 255, 0.1));'
            return f'{color} color: white; font-weight: bold; text-align: center;'

        styled_df = results_df.style.applymap(highlight_score, subset=["Match Score (%)"]).set_properties(**{
            'border-radius': '10px',
            'padding': '10px',
        })

        # Display Results
        st.dataframe(styled_df, use_container_width=True)

        # Beautiful Success Modal
        st.markdown(
            f"""
            <div class="success-modal">
                🎉 Top Match: <strong>{results_df.iloc[0]['Resume']}</strong> with {results_df.iloc[0]['Match Score (%)']}%!
            </div>
            """,
            unsafe_allow_html=True
        )

        # Download Button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="resume_rankings.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.error("⚠️ Please upload resumes and enter a job description!")
else:
    st.info("👆 Upload resumes and a job description, then click 'Rank Resumes Now' to start!")

# Debug message to ensure the app is running
st.write("Debug: App is running. If you don't see the UI, check for CSS conflicts or browser issues.")