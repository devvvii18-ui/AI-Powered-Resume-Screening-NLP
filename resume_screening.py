from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

resume_text = '''
Computer Science graduate with strong skills in Python, SQL, Excel, and Data Analysis.
Experience in data validation, reporting, and problem-solving.
Built projects in machine learning, recommendation systems, and analytics dashboards.
'''

job_description = '''
Looking for a Data Analytics Intern with knowledge of SQL, Python, Excel,
data reporting, dashboard creation, and analytical problem-solving skills.
Experience in machine learning projects is a plus.
'''

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

resume_clean = clean_text(resume_text)
jd_clean = clean_text(job_description)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume_clean, jd_clean])

similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
match_percentage = round(similarity_score * 100, 2)

print("Resume Screening using NLP")
print("-" * 40)
print(f"Resume Match Score: {match_percentage}%")
