
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# STEP 1: Load the dataset

data = pd.read_csv("spam.csv", encoding="latin-1")

# Support common dataset column names
if "v1" in data.columns and "v2" in data.columns:
    data = data[["v1", "v2"]]
    data.columns = ["label", "message"]

elif "label" in data.columns and "message" in data.columns:
    data = data[["label", "message"]]

else:
    raise ValueError(
        "Dataset must contain either v1/v2 or label/message columns."
    )

# Remove missing and duplicate entries
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Keep only valid labels
data = data[data["label"].isin(["ham", "spam"])]

print("Dataset loaded successfully!")
print(data.head())

print("\nDataset distribution:")
print(data["label"].value_counts())


# STEP 2: Prepare the data

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# STEP 3: Build the ML pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("classifier", MultinomialNB())
])


# STEP 4: Train the model

model.fit(X_train, y_train)

print("\nModel training completed!")


# STEP 5: Evaluate the model

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# STEP 6: Save the trained model

joblib.dump(model, "spam_model.pkl")

print("\nModel saved as spam_model.pkl")


# STEP 7: Predict new messages

def predict_message(message):

    prediction = model.predict([message])[0]

    if prediction == "spam":
        return "SPAM"

    return "NOT SPAM"


# STEP 8: Test the model with user input

print("\n--- EMAIL / SMS SPAM DETECTOR ---")
print("Type 'exit' to stop.")

while True:

    message = input("\nEnter your message: ")

    if message.lower() == "exit":
        print("Program closed.")
        break

    result = predict_message(message)

    print("Prediction:", result)
