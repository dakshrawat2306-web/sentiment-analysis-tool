# app.py
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
from flask import Flask, request, jsonify, render_template
from model_vader import predict_vader
from preprocessing import clean_text

app = Flask(__name__)

# This is the API endpoint — a URL that accepts POST requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    raw_text = data['text']
    cleaned = clean_text(raw_text)
    sentiment, confidence = predict_vader(cleaned)

    return jsonify({
        'original_text': raw_text,
        'cleaned_text': cleaned,
        'sentiment': sentiment,
        'confidence': confidence,
        'model_used': 'vader'
    })

# Serve the frontend page
@app.route('/')
def home():
    return render_template('index.html')

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
