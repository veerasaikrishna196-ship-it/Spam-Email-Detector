import streamlit as st
import pickle

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Email Detector")
st.write("Detect whether an email or SMS message is Spam or Not Spam.")

message = st.text_area("✍️ Enter your message")

if st.button("🔍 Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)

        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")