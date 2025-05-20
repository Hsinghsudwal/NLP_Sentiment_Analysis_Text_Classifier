import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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

# Load data from csv
# data='data/simple.csv'
# df = pd.read_csv(data)

# Display the data distribution
sentiment_counts = df['sentiment'].value_counts()
print("Data distribution:")
print(sentiment_counts)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.3, random_state=42
)

# Convert text to numerical features using bag-of-words
vectorizer = CountVectorizer(stop_words='english')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# Make predictions
y_pred = classifier.predict(X_test_vectors)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Plot the sentiment distribution
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()

# Function to predict sentiment of new text
def predict_sentiment(text):
    # Transform the text using the same vectorizer
    text_vector = vectorizer.transform([text])
    
    # Predict sentiment
    sentiment = classifier.predict(text_vector)[0]
    
    # Get probability scores
    proba = classifier.predict_proba(text_vector)[0]
    confidence = max(proba)
    
    return sentiment, confidence

# Interactive section for users to try the model
print("\n--- Sentiment Analysis Prediction ---")
print("Enter a sentence to analyze sentiment (or 'q' to quit):")

while True:
    user_input = input("> ")
    if user_input.lower() == 'q':
        break
    
    sentiment, confidence = predict_sentiment(user_input)
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
