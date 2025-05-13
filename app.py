import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import pickle
import re
import nltk
from typing import Optional, Tuple, Any
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Advanced text preprocessing function
    """
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

class AlexaReviewAnalyzer:
    def __init__(self, base_dir: str = "~/Desktop/sentiment"):
        self.BASE_DIR = os.path.expanduser(base_dir)
        self.MODEL_PATH = os.path.join(self.BASE_DIR, "trained_model.pkl")
        self.VECTORIZER_PATH = os.path.join(self.BASE_DIR, "tfidf_vectorizer.pkl")
        self.DATASET_PATH = os.path.join(self.BASE_DIR, "amazon_alexa.tsv")
        st.set_page_config(page_title="Alexa Reviews Insights", layout="wide")

    def load_data(self) -> Optional[pd.DataFrame]:
        try:
            if not os.path.exists(self.DATASET_PATH):
                st.error(f"Dataset file not found at {self.DATASET_PATH}")
                return None
            
            data = pd.read_csv(self.DATASET_PATH, sep='\t', encoding='utf-8')
            
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y', errors='coerce')

            if 'rating' not in data.columns:
                st.warning("⚠️ 'rating' column missing in dataset. Some visualizations may not work.")
            
            return data
        except Exception as e:
            st.error(f"Dataset Loading Error: {e}")
            return None

    def load_ml_components(self) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load pre-trained model and vectorizer with robust error handling.
        """
        try:
            if not os.path.exists(self.MODEL_PATH):
                st.error(f"Model file missing at: {self.MODEL_PATH}")
                return None, None
            if not os.path.exists(self.VECTORIZER_PATH):
                st.error(f"Vectorizer file missing at: {self.VECTORIZER_PATH}")
                return None, None

            with open(self.MODEL_PATH, "rb") as model_file:
                model = pickle.load(model_file)

            with open(self.VECTORIZER_PATH, "rb") as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)

            if not hasattr(model, "predict"):
                st.error("Loaded model does not have a 'predict' method.")
                return None, None
            if not hasattr(vectorizer, "transform"):
                st.error("Loaded vectorizer does not have a 'transform' method.")
                return None, None

            st.success("✅ Model and Vectorizer Loaded Successfully!")
            return model, vectorizer

        except Exception as e:
            st.error(f"Unexpected Model Loading Error: {e}")
            return None, None

    def sentiment_analyzer(self, model, vectorizer):
        """
        Interactive sentiment analysis module.
        """
        st.title("Review Sentiment Predictor")

        user_input = st.text_area("Enter Alexa Device Review:", 
            "Share your experience with the Alexa device...")

        if st.button("Analyze Sentiment"):
            if user_input.strip():
                try:
                    processed_input = preprocess_text(user_input)
                    input_vector = vectorizer.transform([processed_input])
                    prediction = model.predict(input_vector)[0]

                    sentiment_mapping = {
                        0: ("Negative", "red"),
                        1: ("Positive", "green")
                    }

                    sentiment, color = sentiment_mapping.get(prediction, ("Unknown", "gray"))
                    
                    st.markdown(f"""
                    ### **Predicted Sentiment:**  
                    <span style='color:{color}; font-size:20px; font-weight:bold;'>{sentiment}</span>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Sentiment Analysis Error: {e}")
            else:
                st.warning("Please provide a review to analyze.")

    def run(self):
        """
        Main application execution method.
        """
        st.sidebar.title("Alexa Reviews Explorer")
        app_mode = st.sidebar.radio("Navigation", 
            ["Sentiment Analysis"], index=0)

        model, vectorizer = self.load_ml_components()

        if app_mode == "Sentiment Analysis":
            if model and vectorizer:
                self.sentiment_analyzer(model, vectorizer)
            else:
                st.error("Machine learning model unavailable.")

def main():
    try:
        analyzer = AlexaReviewAnalyzer()
        analyzer.run()
    except Exception as e:
        st.error(f"Unexpected application error: {e}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
