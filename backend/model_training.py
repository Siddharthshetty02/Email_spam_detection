import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("datasets/spam.csv", encoding="latin-1")

# Print column names and first few rows
print("Columns in the dataset:", df.columns.tolist())
print(df.head())

# Select relevant columns and rename them
df = df[['label', 'text']]  # Replace 'v1' and 'v2' with the correct column names
df.columns = ['label', 'text']  # Rename columns for consistency

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Build and train the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, "spam_detection_model.pkl")