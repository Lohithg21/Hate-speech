import re
import nltk
import pandas as pd
import numpy as np
import streamlit as st
import logging
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load dataset
dataset_path = "labeled_data.csv"  # Ensure correct path
try:
    df = pd.read_csv(dataset_path)
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error(f"Dataset not found at {dataset_path}. Please check the path.")
    st.error("Dataset not found! Please check the path.")
    st.stop()

# Rename 'class' column to 'label' if necessary
if 'class' in df.columns:
    df = df.rename(columns={"class": "label"})

# Ensure 'label' column exists
if 'label' not in df.columns:
    logging.error("The dataset does not have a 'label' column. Check the column names!")
    st.error("The dataset does not have a 'label' column!")
    st.stop()

y = df['label']

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokenizer = WhitespaceTokenizer()
    words = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english')) - {"not", "no"}  # Keep negations
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['processed_tweet'] = df['tweet'].apply(preprocess_text)
logging.info("Text preprocessing completed.")

# Convert text to numerical features
vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=2, sublinear_tf=True)
X = vectorizer.fit_transform(df['processed_tweet'])
logging.info("TF-IDF vectorization completed.")

# Balance dataset using SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y)
logging.info("Dataset balancing using SMOTE completed.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)
logging.info("Model training completed.")

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy:.4f}")

# Function to predict hate speech
def predict_hate_speech(text):
    processed_text = preprocess_text(text)
    if not processed_text:
        return "Invalid Input (Empty or Non-textual Data)"
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]  # Get prediction
    
    # Map prediction to correct labels
    label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither (Safe Content)"}
    return label_map.get(prediction, "Unknown")

# ---------------- Streamlit UI ----------------
st.title("🛑 Hate Speech Detection")

st.write("Enter a tweet below to check if it contains hate speech or offensive language.")

user_input = st.text_area("Enter a tweet:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid text!")
    else:
        prediction = predict_hate_speech(user_input)
        st.success(f"Prediction: **{prediction}**")

st.write(f"📊 Model Accuracy: **{accuracy:.2f}**")
