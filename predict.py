import pickle

# Load the saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Take input from the user
message = input("Enter your message: ")

# Convert text to numbers
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

if prediction[0] == 1:
    print("🚨 Spam Message")
else:
    print("✅ Not Spam")