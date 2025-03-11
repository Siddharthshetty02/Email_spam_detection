import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
# Assuming you have a CSV file with columns 'label' and 'text'
data = pd.read_csv('datasets/spam_email_large.csv')

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply preprocessing to the text data
data['text'] = data['text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
import joblib

# Save the model and vectorizer
joblib.dump(model, 'spam_detection_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load the model and vectorizer
model = joblib.load('spam_detection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example usage
new_email = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
new_email_processed = preprocess_text(new_email)
new_email_tfidf = vectorizer.transform([new_email_processed])
prediction = model.predict(new_email_tfidf)

print(f'The email is classified as: {prediction[0]}')