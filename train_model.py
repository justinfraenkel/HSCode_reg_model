import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
df = pd.read_csv('test_data.csv')

# Convert 'HS Code' to numeric if it's categorical (string)
le = LabelEncoder()
df['HS Code'] = le.fit_transform(df['HS Code'])

# Convert all text to lowercase for case insensitivity
df['Product Description'] = df['Product Description'].str.lower()

# Use TF-IDF for text features, also convert to lowercase
tfidf = TfidfVectorizer(max_features=100, lowercase=False)  # We handle case in preprocessing step
X = tfidf.fit_transform(df['Product Description'])
y = df['HS Code']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model, vectorizer, and encoder
joblib.dump((tfidf, model), 'linear_regression_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model trained and saved.")
