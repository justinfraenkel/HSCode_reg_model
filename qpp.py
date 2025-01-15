import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Title of the app
st.title('HS Code Prediction App using Linear Regression')

# Load pre-trained model, vectorizer, and encoder
try:
    tfidf, model = joblib.load('linear_regression_model.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model or encoder not found. Please run the training script first.")
    st.stop()

# User input for prediction
product_description = st.text_input("Enter product description for HS Code prediction:")

if product_description:
    # Convert input to lowercase to match the training data
    product_description = product_description.lower()
    
    # Transform the new description using the same tfidf
    new_description_transformed = tfidf.transform([product_description])
    
    # Predict
    prediction = model.predict(new_description_transformed)
    
    # Adjust prediction to fit within known labels
    predicted_numeric = np.clip(round(prediction[0]), 0, max(le.classes_))  # Clip to min and max of known labels
    
    try:
        # Convert back to original HS code if needed
        predicted_hs_code = le.inverse_transform([int(predicted_numeric)])[0]
        st.write(f"Predicted HS Code: {predicted_hs_code}")
        
        # Feedback form with buttons
        st.subheader("Feedback")
        feedback_choice = st.radio("Is the prediction correct?", options=["Correct", "Incorrect", "Unsure"], index=2)  # Default to "Unsure"
        
        if st.button("Submit Feedback"):
            with open('feedback.csv', 'a') as f:
                f.write(f"{product_description},{predicted_hs_code},{feedback_choice}\n")
            st.success(f"Feedback submitted: {feedback_choice}")
            
    except ValueError as e:
        st.write(f"Error in prediction: {e}")
        st.write("The predicted value was out of range or not in the training data. Consider using a different model or expanding your dataset.")

# Display example data
if st.checkbox('Show example data'):
    st.write(pd.read_csv('test_data.csv'))
