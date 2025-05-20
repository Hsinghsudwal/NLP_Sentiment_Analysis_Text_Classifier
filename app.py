from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

app = Flask(__name__)

# Initialize model variables
vectorizer = None
classifier = None

def train_model():
    """Train the sentiment analysis model and save it"""
    global vectorizer, classifier
    
    # Sample data - you could replace this with a CSV file
    data = {
        'text': [
            "I really enjoyed this movie, it was fantastic!",
            "The product works exactly as described and is great.",
            "Not bad, but could have been better.",
            "This is just okay, nothing special.",
            "The service was terrible and the staff was rude.",
            "I hated every minute of this experience.",
            "Completely disappointed with this purchase.",
            "It works fine for basic needs.",
            "Amazing experience! Would recommend to everyone!",
            "Waste of money, don't buy this."
        ],
        'sentiment': [
            'positive', 'positive', 'neutral', 'neutral', 
            'negative', 'negative', 'negative', 'neutral',
            'positive', 'negative'
        ]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment'], test_size=0.3, random_state=42
    )
    
    # Convert text to numerical features using bag-of-words
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vectors = vectorizer.fit_transform(X_train)
    
    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vectors, y_train)
    
    # Save the models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('models/classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    return vectorizer, classifier

def load_models():
    """Load pre-trained models if they exist"""
    global vectorizer, classifier
    
    try:
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('models/classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
            
        return True
    except:
        return False

def predict_sentiment(text):
    """Predict sentiment of given text"""
    # Transform the text using the vectorizer
    text_vector = vectorizer.transform([text])
    
    # Predict sentiment
    sentiment = classifier.predict(text_vector)[0]
    
    # Get probability scores
    proba = classifier.predict_proba(text_vector)[0]
    confidence = max(proba)
    
    return sentiment, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text from request
        data = request.get_json()
        text = data['text']
        
        # Make prediction
        sentiment, confidence = predict_sentiment(text)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(float(confidence), 2),
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get text from request
        data = request.get_json()
        text = data['text']
        
        # Make prediction
        sentiment, confidence = predict_sentiment(text)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(float(confidence), 2),
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Try to load pre-trained models, or train if they don't exist
    if not load_models():
        vectorizer, classifier = train_model()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
