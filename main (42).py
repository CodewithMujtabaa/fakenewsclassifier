import streamlit as st
import streamlit.components.v1 as components
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
import re
import nltk

nltk.download('stopwords')

model = joblib.load("logistic_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

st.title("📰 Fake News Detection App with Explainability")
st.markdown("Enter a news article or headline to check if it's fake or real.")

user_input = st.text_area("✍️ Paste news text below:")

if st.button("🔍 Predict and Explain"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid news message.")
    else:
        cleaned_input = clean_text(user_input)
        transformed_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_input)[0]
        label = "❌ FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"
        st.subheader(f"🧠 Prediction: {label}")

        st.markdown("---")
        st.markdown("### 🔎 Feature Impact Explanation")

        def predict_proba(texts):
            cleaned_texts = [clean_text(t) for t in texts]
            X = vectorizer.transform(cleaned_texts)
            return model.predict_proba(X)

        masker = shap.maskers.Text()
        explainer = shap.Explainer(predict_proba, masker)
        shap_values = explainer([user_input])

        # Render SHAP explanation as interactive HTML
        components.html(shap.plots.text(shap_values[0], display=False), height=300)
