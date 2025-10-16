import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

TITLE_MAX_LENGTH = 100
TEXT_MAX_LENGTH = 1500

@st.cache_resource
def load_resources():
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    
    with open('encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    
    model = load_model('Model_GRU_FAKE_TRUE.keras')
    
    return tokenizer, encoder, model

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(lemmatized_words)

def process_text_with_tokenizer(text_series, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(text_series.astype(str))
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences

def predict_news(title, text, subject, tokenizer, encoder, model):
    df = pd.DataFrame([{'title': title, 'text': text, 'subject': subject}])
    
    df['text'] = df['text'].apply(lemmatize_text)
    df['title'] = df['title'].apply(lemmatize_text)
    
    padded_title = process_text_with_tokenizer(df['title'], tokenizer, TITLE_MAX_LENGTH)
    padded_text = process_text_with_tokenizer(df['text'], tokenizer, TEXT_MAX_LENGTH)
    
    subject_encoded = encoder.transform(df[['subject']]).astype('float32')
    
    title_input = np.stack(padded_title)
    text_input = np.stack(padded_text)
    
    prediction = model.predict([title_input, text_input, subject_encoded], verbose=0)
    
    return prediction[0][0]

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection System")
st.markdown("*A Machine Learning Based Approach to Identify Misinformation*")

try:
    tokenizer, encoder, model = load_resources()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        title = st.text_input("News Title", placeholder="Enter the news headline...")
        text = st.text_area("News Content", placeholder="Enter the article content...", height=250)
    
    with col2:
        subject = st.selectbox("Category", ["News", "Politics", "Government News", "Left-news", "US_News", "Middle-east", "Politics News"])
        st.write("")
        st.write("")
        predict_btn = st.button("Analyze", type="primary", use_container_width=True)
    
    if predict_btn:
        if not title or not text:
            st.error("Please enter both title and content")
        else:
            with st.spinner("Processing..."):
                score = predict_news(title, text, subject, tokenizer, encoder, model)
                
                st.write("---")
                st.subheader("Results")
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.metric("Confidence", f"{score:.2%}")
                
                with c2:
                    if score >= 0.6:
                        st.success("TRUE NEWS")
                    else:
                        st.error("FAKE NEWS")
                
                with c3:
                    st.write(f"**Fake:** {(1-score)*100:.1f}%")
                    st.write(f"**True:** {score*100:.1f}%")
                
                st.progress(float(score))
                
                if score >= 0.6:
                    st.info("The article appears to be genuine based on language patterns and content structure.")
                else:
                    st.warning("The article shows characteristics of potential misinformation. Verify from credible sources.")
    
    with st.expander("Sample Examples"):
        c1, c2 = st.columns(2)
        
        if c1.button("Fake News Sample"):
            st.write("Title: You Won't Believe Which Celebrity Was Just Replaced By A Robot Clone!")
            st.write("Text: An anonymous insider has leaked documents proving that a beloved Hollywood actor was secretly replaced...")
        
        if c2.button("True News Sample"):
            st.write("Title: Scientists Discover New Species of Deep-Sea Fish")
            st.write("Text: Researchers from NOAA have identified a new species of fish living at depths of over 8,000 meters...")
    
    st.caption("Note: This is a machine learning model. Results should be verified from multiple sources.")

except FileNotFoundError:
    st.error("Model files not found. Please ensure all required files are present.")
except Exception as e:
    st.error(f"Error: {str(e)}")