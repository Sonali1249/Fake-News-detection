from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

app = Flask(__name__,template_folder='./templates',static_folder='./static', static_url_path='/static')

loaded_model = pickle.load(open(os.path.join(os.path.dirname(__file__), "model.pkl"), 'rb'))
vector = pickle.load(open(os.path.join(os.path.dirname(__file__), "vector.pkl"), 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))
corpus = []

def fake_news_det(title, text):
    news = title + " " + text # Combine title and text
    review = news
    review = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    print("Processed input data:", input_data) # Added for debugging
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)[0] # Get the single prediction value
    confidence_score = loaded_model.decision_function(vectorized_input_data)[0] # Get the single confidence score

    return prediction, confidence_score # Return both


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        text = request.form['news']
        pred_label, confidence_score = fake_news_det(title, text) # Unpack the returned values
        print(f"Prediction Label: {pred_label}, Confidence Score: {confidence_score}")

        if pred_label == 1:
            res = "Prediction of the News : Looking Real News📰 "
        else:
            res = "Prediction of the News : Looking Fake News📰"

        # Pass both result and confidence to the template
        return render_template("prediction.html", prediction_text=res, confidence=confidence_score, news_input=text, news_title=title)
    else:
        return render_template('prediction.html', prediction_text="Enter news headline to predict.", news_input="", news_title="")


@app.route('/statistics')
def statistics():
    try:
        df = pd.read_csv('dataset/train.csv')
        total_articles = len(df)
        fake_articles = df[df['label'] == 0].shape[0]
        real_articles = df[df['label'] == 1].shape[0]
        
        # Calculate percentages
        fake_percentage = (fake_articles / total_articles) * 100 if total_articles > 0 else 0
        real_percentage = (real_articles / total_articles) * 100 if total_articles > 0 else 0

        statistics_data = {
            'total_articles': total_articles,
            'fake_articles': fake_articles,
            'real_articles': real_articles,
            'fake_percentage': round(fake_percentage, 2),
            'real_percentage': round(real_percentage, 2)
        }
        return render_template('statistics.html', stats=statistics_data)
    except FileNotFoundError:
        print("Error: train.csv not found. Please make sure the dataset is in the correct directory.")
        return "Error: train.csv not found. Please make sure the dataset is in the correct directory.", 404
    except Exception as e:
        print(f"An error occurred while loading statistics: {e}")
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)