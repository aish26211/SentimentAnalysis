import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

# Load trained model & vectorizer
svm_model = joblib.load("sentiment_shrey1.pkl")
vectorizer = joblib.load("vectorizer1.pkl")

# Predefined results from your model
validation_accuracy = 0.831
classification_report_dict = {
    "Irrelevant": {"precision": 0.78, "recall": 0.75, "f1-score": 0.77, "support": 172},
    "Negative": {"precision": 0.81, "recall": 0.92, "f1-score": 0.87, "support": 266},
    "Neutral": {"precision": 0.90, "recall": 0.78, "f1-score": 0.83, "support": 285},
    "Positive": {"precision": 0.82, "recall": 0.84, "f1-score": 0.83, "support": 277},
    "accuracy": 0.83,
    "macro avg": {"precision": 0.83, "recall": 0.82, "f1-score": 0.82, "support": 1000},
    "weighted avg": {"precision": 0.83, "recall": 0.83, "f1-score": 0.83, "support": 1000}
}

# Confusion matrix example (replace with your actual confusion matrix)
conf_matrix = [
    [129, 16, 23, 4],
    [5, 244, 12, 5],
    [14, 36, 222, 13],
    [5, 11, 23, 238]
]

# Load datasets
def load_dataset(name):
    if name == "Twitter Dataset":
        df = pd.read_csv("twitter_validation.csv", names=["ID", "Product", "Sentiment", "Review"])
        df = df[["Sentiment", "Review"]]
    else:
        df = pd.read_csv("amazon_reviews.csv")
        df = df.rename(columns={"Score": "Sentiment", "Text": "Review"})
        df = df[["Sentiment", "Review"]]
        sentiment_map = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
        df["Sentiment"] = df["Sentiment"].map(sentiment_map)
    return df

# Function to predict sentiment
def predict_sentiment(review):
    input_vector = vectorizer.transform([review])
    return svm_model.predict(input_vector)[0]

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp { background-color: #121212; } /* Dark background */
        [data-testid="stSidebar"] { background-color: #1B2631; }
        [data-testid="stSidebar"] * { color: #FFFFFF !important; font-weight: bold; }
        h1, h2, h3, h4, h5, h6, p, label { color: #FFFFFF; } /* White text for contrast */
        .stButton>button { background-color: #007BFF; color: white; border-radius: 8px; padding: 10px; }
        .stButton>button:hover { background-color: #0056b3; }
        html, body, [class*="st-"] { font-family: 'Segoe UI', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/10106/10106515.png", width=200)
st.sidebar.title("Navigation")
dataset_option = st.sidebar.selectbox("Select Dataset", ["Twitter Dataset", "Amazon Reviews"])
test_df = load_dataset(dataset_option)

page = st.sidebar.radio("Explore", ["Sentiment Analysis", "Dataset Analysis", "Visualizations", "Model Evaluation"])

# Sentiment Analysis Page
if page == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    user_input = st.text_area("Enter your review:", "")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            prediction = predict_sentiment(user_input)
            st.success(f"Predicted Sentiment: {prediction}")
        else:
            st.warning("Please enter a review before analyzing.")

# Dataset Analysis Page
elif page == "Dataset Analysis":
    st.title("Dataset Analysis")
    analysis_option = st.selectbox("Select Analysis Type", ["Sample Data", "Sentiment Distribution", "Summary Statistics"])
    if analysis_option == "Sample Data":
        st.write(test_df.head(10))
    elif analysis_option == "Sentiment Distribution":
        sentiment_counts = test_df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)
    elif analysis_option == "Summary Statistics":
        st.write(test_df.describe(include="all"))

# Visualizations Page
elif page == "Visualizations":
    st.title("Data Visualizations")
    st.subheader("Sentiment Distribution in Selected Dataset")
    fig, ax = plt.subplots()
    sentiment_counts = test_df["Sentiment"].value_counts()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["red", "green", "blue"])
    ax.axis("equal")
    st.pyplot(fig)

    st.subheader("WordCloud for Sentiments")
    sentiment_option = st.selectbox("Select Sentiment", ["Positive", "Negative", "Neutral"])
    def generate_wordcloud(sentiment):
        text = " ".join(review for review in test_df[test_df["Sentiment"] == sentiment]["Review"].dropna())
        return WordCloud(width=600, height=400, background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(generate_wordcloud(sentiment_option), interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("Model Evaluation Results")
    st.subheader("Validation Accuracy")
    st.write(f"Validation Accuracy: {validation_accuracy}")

    st.subheader("Classification Report")
    classification_df = pd.DataFrame(classification_report_dict).transpose()
    st.dataframe(classification_df)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Irrelevant", "Negative", "Neutral", "Positive"], yticklabels=["Irrelevant", "Negative", "Neutral", "Positive"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Aish Sinha")
