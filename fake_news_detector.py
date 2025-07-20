import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("news.csv")  # Ensure this file exists
df = df[['text', 'label']]

# Clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

df['text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f" Model Accuracy: {round(accuracy * 100, 2)}%")
print(" Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Manual test ---
test_news = [
    "Breaking: Aliens have landed in London!",
    "The government passed a new education reform bill."
]

test_news_cleaned = [clean_text(news) for news in test_news]
test_news_tfidf = vectorizer.transform(test_news_cleaned)
predictions = model.predict(test_news_tfidf)

for i, news in enumerate(test_news):
    print(f"\n News: {news}")
    print(f" Prediction: {predictions[i]}")

# --- Interactive user input ---
while True:
    user_input = input("\n Enter a news headline (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print(" Exiting prediction loop.")
        break
    cleaned_input = clean_text(user_input)
    input_tfidf = vectorizer.transform([cleaned_input])
    result = model.predict(input_tfidf)[0]
    print(f" Prediction: {result}")
