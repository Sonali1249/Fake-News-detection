import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Load model and vectorizer
@st.cache_resource
def load_resources():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    vector_path = os.path.join(os.path.dirname(__file__), "vector.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vector_path, 'rb') as f:
        vector = pickle.load(f)
    
    # Download NLTK data
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        
    return model, vector

loaded_model, vector = load_resources()
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def fake_news_det(title, text):
    news = title + " " + text
    review = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', news)
    review = review.lower()
    review = nltk.word_tokenize(review)
    
    corpus = []
    for y in review:
        if y not in stpwrds:
            corpus.append(lemmatizer.lemmatize(y))
    
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)[0]
    confidence_score = loaded_model.decision_function(vectorized_input_data)[0]
    
    return prediction, confidence_score

# UI Design
st.title("📰 Fake News Detector")
st.markdown("### Made by Sonali Tiwari")
st.write("Enter the news title and content below to check if it's potentially fake or real.")

title = st.text_input("News Title", placeholder="Enter the headline here...")
text = st.text_area("News Content", placeholder="Paste the news article content here...", height=200)

if st.button("Predict"):
    if title and text:
        with st.spinner('Analyzing...'):
            pred_label, confidence = fake_news_det(title, text)
            
            st.divider()
            if pred_label == 1:
                st.success("### Prediction: Looking Real News 📰")
            else:
                st.error("### Prediction: Looking Fake News 📰")
            
            st.info(f"**Confidence Score:** {confidence:.4f}")
            st.write("---")
            st.write("**Processed input overview:**")
            st.caption(f"Title: {title}")
    else:
        st.warning("Please enter both a title and news content.")

# Sidebar for additional info
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Passive Aggressive Classifier model to detect fake news "
    "with an accuracy of approximately 96%."
)
st.sidebar.markdown("---")
st.sidebar.write("### Project Source")
st.sidebar.write("[GitHub Repository](https://github.com/Sonali1249/Fake-News-detection.git)")
