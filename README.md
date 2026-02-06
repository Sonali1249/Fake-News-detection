# Fake News Detection using Machine Learning
**Made by Sonali Tiwari**

## Introduction
This repository contains a comprehensive project for detecting fake news using machine learning techniques and various natural language processing techniques. The project includes data analysis, model training, and a web application for real-time fake news detection. The machine learning model is designed to classify news articles as either real or fake based on their content.

## Problem Definition
We aim to develop a machine learning program to identify when a news source may be producing fake news. The model will focus on identifying fake news sources, based on multiple articles originating from a source. Once a source is labeled as a producer of fake news, we can predict with high confidence that any future articles from that source will also be fake news. Focusing on sources widens our article misclassification tolerance, because we will have multiple data points coming from each source.

The intended application of the project is for use in applying visibility weights in social media. Using weights produced by this model, social networks can make stories that are highly likely to be fake news less visible.

## Project Structure
The repository is organized into the following directories and files:
- **Images**: Contains important project images, such as block diagrams, classification reports, confusion matrices, and screenshots.
- **dataset**: Includes the dataset, consisting of train and test data from Kaggle, which is used to train and test the model.
- **static**: Houses static assets for the web application, including CSS, JavaScript, etc.
- **templates**: Includes HTML templates for the web application, such as `Landingpage.html` and `prediction page.html`.
- **Fake_News_Detector-PA.ipynb**: Jupyter Notebook file for data analysis and model training.
- **app.py**: Flask web application for real-time fake news detection.
- **model.pkl**: Pre-trained machine learning model for fake news detection.
- **vector.pkl**: Pre-trained vectorizer for text data.

## Datasets 
### train.csv
A full training dataset with the following attributes:
- `id`: unique id for a news article
- `title`: the title of a news article
- `author`: author of the news article
- `text`: the text of the article; could be incomplete
- `label`: a label that marks the article as potentially unreliable
  - `1`: unreliable
  - `0`: reliable

### test.csv
A testing training dataset with all the same attributes as `train.csv` without the label.

## Installation

To run this project, you need to have Python installed on your system. Follow these steps to set up the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sonali1249/Fake-News-detection.git
   cd Fake-News-detection
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   The project uses NLTK for text processing. You may need to download the necessary datasets by running:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

## How to Run

1. **Start the Flask Application**:
   Run the following command in your terminal:
   ```bash
   python app.py
   ```

2. **Access the Web App**:
   Once the server is running, open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## How to Run with Streamlit

Streamlit provides a very clean and modern UI. To run the Streamlit version:

1. **Start the Streamlit App**:
   Run the following command in your terminal:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the App**:
   The app will automatically open in your default browser, usually at `http://localhost:8501`.

## How to Deploy on Streamlit Cloud

You can deploy this project for free on [Streamlit Cloud](https://streamlit.io/cloud) by following these steps:

1. **GitHub Setup**:
   - Push your code to your GitHub repository: `https://github.com/Sonali1249/Fake-News-detection.git`.
   - Ensure `requirements.txt`, `streamlit_app.py`, `model.pkl`, and `vector.pkl` are in the root directory.

2. **Streamlit Cloud**:
   - Log in to Streamlit Cloud and click **"New app"**.
   - Select your repository and the main branch.
   - Set the main file path to `streamlit_app.py`.
   - Click **"Deploy!"**.

3. **NLTK Handling**:
   The `streamlit_app.py` script is designed to automatically download the necessary NLTK data upon startup, so no extra configuration is needed for the NLTK datasets on Streamlit Cloud.

## How to Check Fake News

1. **Navigate to the Prediction Page**: Click on the "Predict" link or button on the home page.
2. **Enter News Details**:
   - **Title**: Enter the headline of the news article.
   - **News Content**: Paste the main text of the article.
3. **Submit**: Click the "Submit" or "Predict" button.
4. **View Results**: The application will display whether the news is **Real** or **Fake** along with a confidence score.

## Sample News for Testing

You can use these examples to test the application:

### Example 1: Real News 📰
- **Title**: NASA's James Webb Telescope Captures Stunning Image of Distant Galaxy
- **Content**: The James Webb Space Telescope has provided its latest high-resolution image, revealing intricate details of a galaxy billions of light-years away. NASA scientists say this data will help understand the early formation of the universe.

### Example 2: Fake News 📰
- **Title**: Scientists Discover Secret Island Where Dinosaurs Still Exist
- **Content**: A team of undercover explorers has reportedly found a hidden island in the Pacific Ocean where prehistoric dinosaurs are still alive and roaming freely. The government is allegedly keeping this a secret from the public.

## Statistics

You can also view the dataset statistics by navigating to `http://127.0.0.1:5000/statistics`, which shows the distribution of real vs. fake news articles in the training dataset.

## Model Name
The machine learning model used for fake news detection in this project is the **Passive Aggressive Classifier**.

### Model Description
The Passive Aggressive Classifier (PAC) is a type of online learning algorithm for binary classification tasks. It is well-suited for applications like fake news detection. The PAC algorithm updates its model continuously as new data arrives, making it efficient for real-time classification.

### Model Accuracy
The Passive Aggressive Classifier achieved an impressive accuracy of **96%** during evaluation. This high accuracy indicates its effectiveness in classifying news articles as reliable or unreliable.

The model is pre-trained and available as `model.pkl` in this repository, allowing you to use it for making predictions.

Feel free to explore the Jupyter Notebook (`Fake_News_Detector-PA.ipynb`) for more details about the model's training and performance.

