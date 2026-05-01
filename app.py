import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

st.set_page_config(
    page_title="AI-Powered Resume Screening",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI-Powered Resume Screening using NLP")
st.markdown(
    "Compare a **Resume** with a **Job Description** using "
    "**TF-IDF Vectorization + Cosine Similarity** and calculate the match score."
)

st.write("---")

resume_text = st.text_area(
    "📌 Paste Resume Text",
    height=220,
    placeholder="Paste resume content here..."
)

job_description = st.text_area(
    "📌 Paste Job Description",
    height=220,
    placeholder="Paste job description here..."
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

if st.button("🚀 Analyze Match Score"):
    if not resume_text or not job_description:
        st.warning("⚠ Please enter both Resume Text and Job Description.")
    else:
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_description)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])

        similarity_score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        match_percentage = round(similarity_score * 100, 2)

        st.write("---")
        st.subheader("📊 Match Result")
        st.progress(int(match_percentage))
        st.success(f"Resume Match Score: {match_percentage}%")

        if match_percentage >= 80:
            st.success("🏆 Excellent Match! Strong fit for the role.")
        elif match_percentage >= 60:
            st.info("👍 Good Match! Resume aligns well.")
        else:
            st.error("⚠ Needs Improvement. Add more relevant skills and keywords.")

        st.write("---")
        st.subheader("💡 Key Insights")

        st.markdown("""
- Relevant skills improve the match score significantly  
- Keywords from the job description matter a lot  
- Better alignment increases recruiter shortlisting chances  
- This project simulates basic ATS (Applicant Tracking System) behavior  
        """)

        st.write("---")
        st.caption("Built by Devi Shankar 🚀 | NLP + Machine Learning Project")
