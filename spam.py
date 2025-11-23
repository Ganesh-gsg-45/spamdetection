import streamlit as st 
import pandas as pd 
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure stopwords are available (quiet)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Helper to load pickles with error handling
def load_pickle(path, name):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f'Failed to load {name} from {path}: {e}')
        return None

# Load vectorizer and model
tfidf = load_pickle('tfidf.pkl', 'TF-IDF vectorizer')
# `bow.pkl` is not used by this app — skip loading to avoid unused variable
model = load_pickle('random.pkl', 'Model')

if tfidf is None or model is None:
    st.stop()

# Initialize stemmer and stopwords list
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Remove non-alphabet characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords, apply stemming
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit app UI
st.title('Spam Detection')
st.write('This is a simple email spam detection app using machine learning algorithms.')

message = st.text_area('Enter your message:', height=150)

if st.button('Predict'):
    if not message or not message.strip():
        st.warning('Please enter a message to classify.')
    else:
        processed_message = preprocess_text(message)
        try:
            vectorized_message = tfidf.transform([processed_message])
            prediction = model.predict(vectorized_message)[0]
            result = 'Not Spam' if int(prediction) == 1 else 'Spam'
            st.success(f'Prediction: {result}')
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(vectorized_message)[0].max()
                st.info(f'Confidence: {proba:.2f}')
        except Exception as e:
            st.error(f'Error during prediction: {e}')
