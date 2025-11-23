# Spam vs Ham Classification using BOW, TF-IDF, and Machine Learning

## Project Overview
This project demonstrates a text classification task to identify whether an SMS message is spam or not (ham). It uses Natural Language Processing (NLP) techniques for text preprocessing and feature extraction, followed by machine learning models to classify messages.

The main techniques utilized include:
- **Bag of Words (BOW)** with CountVectorizer
- **TF-IDF (Term Frequency-Inverse Document Frequency)** Vectorization
- **Random Forest Classifier** for spam/ham classification

## Dataset
The dataset used is `SMSSpamCollection.txt` which contains labeled SMS messages as spam or ham. It is located in the `data/` directory.

## Implementation Details

### Notebook: `spamdetection.ipynb`
- Loads and preprocesses the dataset with steps like:
  - Text cleaning (removal of non-alphabet characters)
  - Lowercasing
  - Stopwords removal
  - Stemming using Porter Stemmer
- Extracts features with CountVectorizer (BOW) and TfidfVectorizer.
- Trains a Random Forest Classifier on both feature sets.
- Evaluates model performance with classification reports and accuracy scores.
- Saves vectorizers and trained model as pickle files (`bow.pkl`, `tfidf.pkl`, `random.pkl`).

### Application: `gsg.py`
- A Streamlit-based web app interface for interacting with the trained model.
- Allows users to input an SMS message.
- Preprocesses the input message using the same steps as in training.
- Uses the saved TF-IDF vectorizer and trained model to predict spam or ham.
- Displays the prediction result ("Spam" or "Not Spam") in the app.

## How to Run the Project

### Prerequisites
- Python 3.x installed
- Packages: pandas, numpy, scikit-learn, nltk, streamlit, and others as used in the notebook/app.

### Steps
1. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn nltk streamlit
   ```
2. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
3. Run the notebook `spamdetection.ipynb` to train and save the model and vectorizers, or use the pre-saved pickle files.
4. Start the Streamlit app:
   ```bash
   streamlit run gsg.py
   ```
5. Enter your message in the app to get a spam/ham prediction.

## File Structure
```
├── data/
│   └── SMSSpamCollection.txt       # Dataset file
├── bow.pkl                        # Saved Bag of Words vectorizer
├── tfidf.pkl                      # Saved TF-IDF vectorizer
├── random.pkl                     # Saved Random Forest model
├── gsg.py                        # Streamlit app for spam detection
├── spamdetection.ipynb           # Jupyter notebook for model creation
└── README.md                     # This readme file
```

## Acknowledgments
- Dataset source: [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Libraries: scikit-learn, nltk, streamlit

---

This project serves as a practical introduction to Natural Language Processing and machine learning for text classification.
