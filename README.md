 CUSTOMER CHURN  PREDICTION  DASHBOARD

This Streamlit app is an interactive web-based tool that helps telecom businesses predict customer churn using machine learning and sentiment analysis. Users can upload customer data in CSV format, and the app will process it, predict which customers are likely to churn, and analyze their feedback sentiment (positive, neutral, or negative).



 WHAT THE APP DOES

  Predicts Customer Churn: Based on structured data such as monthly charges, contract type, internet service, etc., the app uses a pre-trained XGBoost model to predict whether a customer is likely to leave ("churn") or stay.

  Analyzes Customer Feedback: If customer feedback is available (or simulated), it is cleaned and transformed using TF-IDF (Term Frequency-Inverse Document Frequency) and processed by the model for prediction.

  Performs Sentiment Analysis: The app uses TextBlob to evaluate the sentiment of customer feedback and visualizes how sentiment correlates with churn likelihood.

  Displays Interactive Results:
  - Churn predictions for each customer
  - Overall churn rate
  - Sentiment distribution across churned vs. retained customers



 TECHNOLOGIES USED

 Component             Library Used                     
|-------------------|----------------------------------|
| Frontend/UI       | Streamlit                        |
| ML Model          | XGBoost                          |
| Text Vectorization| scikit-learn's TfidfVectorizer   |
| Sentiment Analysis| TextBlob                         |
| Data Processing   | pandas, numpy                    |
| Visualizations    | seaborn, matplotlib              |
| Model Persistence | joblib                           |
| Text Cleaning     | NLTK (stopwords)                 |



 📁 Folder Structure
churn-prediction-app/
├── app.py # Main Streamlit app
├── churn_model.pkl # Trained XGBoost model
├── tfidf_model.pkl # Trained TF-IDF vectorizer
├── training_columns.pkl # List of columns used during training
├── requirements.txt # List of dependencies
├── README.md # Project documentation
└── sample_data.csv # Optional sample input


