# Sentiment Analysis 

A simple sentiment analysis implementation using
scikit-learn and NLP techniques. It classifies text as positive, 
negative, or neutral based on a training dataset. Also create a simple web application for sentiment analysis built with Flask

## Features

- Text sentiment classification (positive, negative, neutral)
- Web interface for interactive use
- REST API for integration with other applications
- Docker support for easy deployment

## Installation and Running

### Option 1: Running with Python

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and navigate to `http://localhost:5000`

### Option 2: Running with Docker

1. Build the Docker image:
   ```
   docker build -t sentiment-analysis-app .
   ```
2. Run the container:
   ```
   docker run -p 5000:5000 sentiment-analysis-app
   ```
3. Open your browser and navigate to `http://localhost:5000`

## API Usage

The application provides a REST API that can be used to programmatically analyze text:

### Endpoint: `/api/predict`

**Method**: POST

**Request Body**:
```json
{
  "text": "Your text to analyze"
}
```

**Response**:
```json
{
  "sentiment": "positive",
  "confidence": 0.85,
  "text": "Your text to analyze"
}
```

## Example API Usage

```python
import requests
import json

url = "http://localhost:5000/api/predict"
data = {"text": "I really enjoyed this movie!"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
result = response.json()

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

## Model Information

The sentiment analysis uses a Multinomial Naive Bayes classifier trained on a small dataset. The text is processed using a bag-of-words approach with CountVectorizer.



## Extending the Project

- Add more training data for better accuracy
- Implement more advanced NLP techniques (TF-IDF, word embeddings)
- Add support for multiple languages
- Include more detailed sentiment analysis (e.g., emotion detection)
