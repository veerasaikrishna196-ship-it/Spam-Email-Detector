import pandas as pd

# Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only first two columns
data = data.iloc[:, :2]

# Rename columns
data.columns = ["label", "message"]

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

print(data.head())
print(data.shape)
from sklearn.model_selection import train_test_split

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
from sklearn.feature_extraction.text import CountVectorizer

# Convert text into numbers
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Vectorization Completed!")
from sklearn.naive_bayes import MultinomialNB

# Create and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model Trained Successfully!")
from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
# Test with your own message
message = ["Hi Krishna, are you coming to college today?"]
message_vector = vectorizer.transform(message)
prediction = model.predict(message_vector)

if prediction[0] == 1:
    print("Spam")
else:
    print("Not Spam")
    import pickle

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))

# Save the vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and Vectorizer saved successfully!")