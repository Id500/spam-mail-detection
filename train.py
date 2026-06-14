import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Kaggle SMS Spam Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")


# Keep required columns
df = df[['v1','v2']]


# Rename columns
df.columns = ['label','message']


# Convert labels
# ham = 0 (not spam)
# spam = 1
df['label'] = df['label'].map({
    'ham':0,
    'spam':1
})


# Remove empty values
df.dropna(inplace=True)


# Input and output
X = df['message']
y = df['label']


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Convert text to numbers
vectorizer = TfidfVectorizer(
    stop_words='english'
)


X_train_vector = vectorizer.fit_transform(X_train)

X_test_vector = vectorizer.transform(X_test)


# Train model
model = LogisticRegression()


model.fit(
    X_train_vector,
    y_train
)


# Test accuracy
prediction = model.predict(X_test_vector)


accuracy = accuracy_score(
    y_test,
    prediction
)


print("Accuracy:", accuracy)


# Save model
joblib.dump(
    model,
    "model.pkl"
)


joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)


print("Training completed")