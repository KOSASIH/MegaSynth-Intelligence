import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Load and preprocess the data
data = [
    ("I love this movie", "positive"),
    ("This movie is great", "positive"),
    ("I don't like this movie", "negative"),
    ("This movie is terrible", "negative")
]

preprocessed_data = [(preprocess_text(text), label) for text, label in data]

# Feature extraction
vectorizer = CountVectorizer()  # or TfidfVectorizer()
X = vectorizer.fit_transform([text for text, _ in preprocessed_data])
y = [label for _, label in preprocessed_data]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the machine learning model
model = MultinomialNB()  # or SVC()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')

# Print performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Sample sentiment analysis
sample_text = "This movie is amazing!"
preprocessed_sample_text = preprocess_text(sample_text)
sample_vector = vectorizer.transform([preprocessed_sample_text])
sentiment = model.predict(sample_vector)[0]

print("Sample Text:", sample_text)
print("Sentiment:", sentiment)
