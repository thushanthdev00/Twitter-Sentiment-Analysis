"""
Twitter Sentiment Analysis - Streamlit App

Loads models/trained_model.sav (LogisticRegression) and
models/vectorizer.sav (TfidfVectorizer) produced by train_model.py,
and predicts whether a tweet / short text is Positive or Negative.
"""

import pickle
import re
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="", layout="centered")


@st.cache_resource
def load_nltk_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "trained_model.sav", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "vectorizer.sav", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


STOPWORDS = load_nltk_stopwords()


@st.cache_resource
def get_stemmer():
    from nltk.stem.porter import PorterStemmer
    return PorterStemmer()


port_stem = get_stemmer()


def stemming(content: str) -> str:
    stem_content = re.sub("[^a-zA-Z]", " ", content)
    stem_content = stem_content.lower()
    stem_content = stem_content.split()
    stem_content = [port_stem.stem(word) for word in stem_content if word not in STOPWORDS]
    return " ".join(stem_content)


def predict_sentiment(text: str, model, vectorizer):
    cleaned = stemming(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    label = "Positive" if prediction == 1 else "Negative"
    return label, confidence


# ---- UI ----
st.title("Twitter Sentiment Analysis")
#st.caption("Logistic Regression + TF-IDF, trained on the Sentiment140 dataset (1.6M tweets)")

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` first to generate "
        "`models/trained_model.sav` and `models/vectorizer.sav`."
    )
    st.stop()

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab1:
    text_input = st.text_area(
        "Enter a tweet or short text",
        placeholder="e.g. I absolutely loved the new update, works great!",
        height=200,
    )

    if st.button("Predict Sentiment", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            label, confidence = predict_sentiment(text_input, model, vectorizer)
            if label == "Positive":
                st.success(f"**Positive** 😊  (confidence: {confidence:.1%})")
            else:
                st.error(f"**Negative** 😞  (confidence: {confidence:.1%})")

with tab2:
    st.write("Upload a CSV with a column named **text** to get predictions for many rows at once.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            with st.spinner("Predicting..."):
                cleaned = df["text"].astype(str).apply(stemming)
                vectors = vectorizer.transform(cleaned)
                preds = model.predict(vectors)
                probs = model.predict_proba(vectors)
                df["sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]
                df["confidence"] = [probs[i][preds[i]] for i in range(len(preds))]

            st.dataframe(df[["text", "sentiment", "confidence"]])
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv, "sentiment_predictions.csv", "text/csv")

st.divider()
with st.expander("About this model"):
    st.markdown(
        """
        - **Dataset:** Sentiment140 (1.6M labeled tweets)
        - **Preprocessing:** regex cleanup → lowercase → stopword removal → Porter stemming
        - **Features:** TF-IDF vectorization
        - **Model:** Logistic Regression
        """
    )
