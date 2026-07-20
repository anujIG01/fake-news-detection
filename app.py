import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load("models/logistic_regression_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict_news(text):
    vectorized = tfidf.transform([clean_text(text)])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0][prediction] * 100
    label = "Real" if prediction == 1 else "Fake"
    return label, confidence

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("<h1 style='text-align:center; margin-bottom:0;'>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Paste a news article below and find out if it's real or fake.</p>", unsafe_allow_html=True)
st.divider()

user_input = st.text_area(
    "News Article Text",
    height=130,
    placeholder="Paste the full article text here...",
    label_visibility="collapsed"
)

word_count = len(user_input.split())
st.caption(f"{word_count} words")

check_clicked = st.button("🔍 Check News", use_container_width=True, type="primary")

if check_clicked:
    if user_input.strip() == "":
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Analyzing article..."):
            label, confidence = predict_news(user_input)

        st.divider()

        if label == "Real":
            st.success(f"✅ This looks like **Real News**")
        else:
            st.error(f"🚨 This looks like **Fake News**")

        st.write(f"**Confidence: {confidence:.1f}%**")
        st.progress(int(confidence))