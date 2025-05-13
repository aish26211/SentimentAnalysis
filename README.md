# 📊 Sentiment Analysis Dashboard

An interactive sentiment analysis web app built using **Python**, **Scikit-learn**, and **Streamlit** that classifies reviews into **Positive**, **Negative**, **Neutral**, and **Irrelevant** sentiments. The model is trained on **Twitter product reviews** and validated against both **Twitter** and **Amazon** review datasets.

---

## 🔍 Features

- 🧹 **Text Preprocessing** using TF-IDF vectorization
- 🤖 Sentiment Classification using **SVM (Support Vector Machine)**
- 📈 Accuracy:  
  - Twitter Dataset: **89%**  
  - Amazon Dataset: **86%**
- 📊 Real-time **data visualization** with **Matplotlib** and **Seaborn**
- 🔄 Interactive Streamlit Dashboard with dataset selection dropdown
- 🧪 Model evaluated using **Precision, Recall, F1-Score**

---

## 🗂️ Project Folder Structure

```bash
sentiment/
├── app1.py                    # Streamlit dashboard application
├── twitter_dataset.csv        # Main dataset (Twitter reviews)
├── amazon_reviews.csv         # Comparative dataset (Amazon reviews)
├── sentiment_model.pkl        # Trained SVM model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt           # Required Python packages
├── README.md                  # Project documentation
└── assets/
    └── dashboard-screenshot.png   # Dashboard screenshot (optional)
``` 


---

## 📊 Datasets Used

| Dataset       | Source           | Purpose                          |
|---------------|------------------|----------------------------------|
| Twitter Reviews | Kaggle (custom) | Training and testing sentiment model |
| Amazon Reviews | Kaggle           | Comparative dataset for analysis |

---

## 🧠 Model & Metrics

| Metric     | Twitter Dataset | Amazon Dataset |
|------------|-----------------|----------------|
| Accuracy   | 89%             | 86%            |
| Precision  | 88%             | 85%            |
| Recall     | 87%             | 84%            |
| F1-Score   | 87.5%           | 84.5%          |

Model: **SVM (Support Vector Machine)**  
Vectorizer: **TF-IDF (Term Frequency-Inverse Document Frequency)**

---

## 🚀 Running the App

### 🔧 Step 1: Install Dependencies

```bash
pip install -r requirements.txt
streamlit run app1.py
``` 

Open your browser and go to: http://localhost:8501


## 🛠️ Technologies Used

- **Python**
- **Scikit-learn**
- **Streamlit**
- **Matplotlib** / **Seaborn**
- **Pandas** / **NumPy**

---

## 💡 Future Improvements

- 🚀 Deploy to **Streamlit Cloud**
- 🐦 Add **real-time Twitter scraping**
- 🤖 Explore transformer-based models like **BERT** for improved accuracy

---

## 🙌 Acknowledgements

- 📊 **Twitter Dataset** — *https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis*  
- 🛒 **Amazon Product Reviews** — *https://www.kaggle.com/datasets/bittlingmayer/amazonreviews*
