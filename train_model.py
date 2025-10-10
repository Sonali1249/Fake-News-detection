import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset
import os
train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset", "train.csv"))

# Drop rows with missing values
train_df.dropna(inplace=True)

# Combine 'title' and 'text' columns for better feature representation
train_df['full_text'] = train_df['title'] + ' ' + train_df['text']

# Separate labels
labels = train_df['label']
X = train_df['full_text']

# Initialize WordNetLemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    review = re.sub(r'[^a-zA-Z\s]', '', text)
    review = review.lower()
    review = nltk.word_tokenize(review)
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    return ' '.join(review)

# Apply preprocessing to the combined text
X = X.apply(preprocess_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

# Initialize and fit TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize and train PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Save the model and vectorizer
with open('model.pkl', 'wb') as handle:
    pickle.dump(pac, handle)

with open('vector.pkl', 'wb') as handle:
    pickle.dump(tfidf_vectorizer, handle)

print("Model and vectorizer saved successfully.")