# import packages 
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Loading the saved files
lg = pickle.load(open('logistic_regression.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('llabel_encoder.pkl', 'rb'))

# Cleaning the text
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


# Prediction function
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(lg.predict(input_vectorized))
    
    return predicted_emotion, label


# Creating APP
st.markdown("<h1 style='font-size: 40px;'>Six Human Emotions Detection APP</h1>", unsafe_allow_html=True)
st.write("---------------------------------------------------------")
st.write("['Joy', 'Fear', 'Anger', 'Love', 'Sadness', 'Surprise']")
st.write("---------------------------------------------------------")

# Taken input from user
user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    predicted_emotion, label = predict_emotion(user_input)
    st.write("Predicted Emotion: ", predicted_emotion)
    st.write("Probability: ", label)
    
