import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model, TF-IDF vectorizer, and training columns
try:
    model = joblib.load("churn_model.pkl")
    tfidf_for_streamlit = joblib.load("tfidf_model.pkl")
    training_columns_streamlit = joblib.load("training_columns.pkl")
except FileNotFoundError:
    st.error("Model files not found.")
    st.stop()

# Download stopwords if missing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words_streamlit = set(stopwords.words('english'))

def clean_text_streamlit(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = [word for word in text.split() if word not in stop_words_streamlit]
    return ' '.join(tokens)

def preprocess_streamlit(df_input):
    df = df_input.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')

    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Add simulated feedback if not present
    if 'Customer_Feedback' not in df.columns:
        np.random.seed(42)
        feedback_samples = [
            "Great service, no complaints.",
            "Very disappointed with the network.",
            "Fast internet and reliable billing.",
            "Unhelpful staff and inconsistent connection.",
            "Excellent value for the price."
        ]
        df['Customer_Feedback'] = np.random.choice(feedback_samples, size=len(df))

    df['Clean_Feedback'] = df['Customer_Feedback'].apply(clean_text_streamlit)

    tfidf_matrix_new = tfidf_for_streamlit.transform(df['Clean_Feedback'])
    tfidf_df_new = pd.DataFrame(tfidf_matrix_new.toarray(), columns=tfidf_for_streamlit.get_feature_names_out())
    tfidf_df_new.index = df.index

    drop_cols = ['customerID', 'Churn', 'Customer_Feedback', 'Clean_Feedback']
    for col in drop_cols:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)

    X_final = pd.concat([df_processed, tfidf_df_new], axis=1)

    missing_cols = set(training_columns_streamlit) - set(X_final.columns)
    for c in missing_cols:
        X_final[c] = 0

    extra_cols = set(X_final.columns) - set(training_columns_streamlit)
    for c in extra_cols:
        X_final = X_final.drop(labels=c, axis=1)

    X_final = X_final[training_columns_streamlit]

    # Return X_final and the cleaned raw data (used for display)
    df_display = df.loc[X_final.index].copy()  # ensure same row count
    return X_final, df_display

# Streamlit UI
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Upload customer data and get churn predictions + feedback insights.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head())

    with st.spinner("Processing data and predicting churn..."):
        X_final, df_display = preprocess_streamlit(df_raw)
        preds = model.predict(X_final)
        df_display['Predicted_Churn'] = preds
        df_display['Predicted_Churn'] = df_display['Predicted_Churn'].map({1: "Yes", 0: "No"})

    st.subheader(" Prediction Results")
    if 'customerID' in df_display.columns:
        st.dataframe(df_display[['customerID', 'Predicted_Churn']])
    else:
        st.dataframe(df_display[['Predicted_Churn']])

    churn_rate = round((preds.sum() / len(preds)) * 100, 2)
    st.success(f"**Overall Churn Rate:** {churn_rate}% of customers are likely to leave.")

    if 'Customer_Feedback' in df_display.columns:
        st.subheader("Customer Feedback Sentiment (Sample)")
        df_display['Clean_Feedback_Display'] = df_display['Customer_Feedback'].apply(clean_text_streamlit)
        df_display['Sentiment'] = df_display['Clean_Feedback_Display'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        ).apply(lambda p: 'positive' if p > 0.2 else ('negative' if p < -0.2 else 'neutral'))

        sentiment_churn_df = df_display.groupby(['Sentiment', 'Predicted_Churn']).size().unstack(fill_value=0)
        st.dataframe(sentiment_churn_df)

        st.markdown("""
            <style>
            .big-font {
                font-size:20px !important;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
        st.markdown("<p class='big-font'>Sentiment Breakdown by Predicted Churn:</p>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_display, x='Sentiment', hue='Predicted_Churn', palette='viridis', ax=ax)
        ax.set_title('Predicted Churn Status by Customer Sentiment')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Number of Customers')
        st.pyplot(fig)
