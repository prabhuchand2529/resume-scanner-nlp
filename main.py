import os 
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
from nltk.corpus import stopwords
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    filtered_words = [w for w in words if w not in stopwords.words('english')]

    return " ".join(filtered_words)

# Read Job Description
with open("job_description.txt", "r") as file:
    job_desc = file.read()

job_desc = preprocess(job_desc)

# Read Resumes
resumes = []
resume_names = []

for filename in os.listdir("resumes"):
    with open(f"resumes/{filename}", "r") as file:
        text = file.read()
        resumes.append(preprocess(text))
        resume_names.append(filename)

#TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([job_desc] + resumes)

#Similarity
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

#Output
print("\n Resume Matching Scores:\n")

for i, score in enumerate(similarities[0]):
    print(resume_names[i], ":", round(score * 100, 2), "% match")