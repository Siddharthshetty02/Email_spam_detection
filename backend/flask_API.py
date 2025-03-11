from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("spam_detection_model.pkl")

# Route for the root URL
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the email text from the request
    data = request.json
    email_text = data['text']

    # Predict using the model
    prediction = model.predict([email_text])
    result = "Spam" if prediction[0] == 1 else "Legit"

    # Return the result
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)