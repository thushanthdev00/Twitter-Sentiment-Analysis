import re
import pickle
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords")

# ---- Paths (Colab-style, adjust if running locally) ----
BASE_DIR = Path(r"T:\Twitter sentiment analysis\twitter-sentiment-analysis")
DATASET_ZIP = BASE_DIR / "sentiment140.zip"
DATA_PATH = BASE_DIR / "training.1600000.processed.noemoticon.csv"
MODELS_DIR = BASE_DIR / "models"

STOPWORDS = set(stopwords.words("english"))
port_stem = PorterStemmer()


def extract_dataset():

    if DATA_PATH.exists():
        print(f"Dataset already extracted at {DATA_PATH}")
        return

    if not DATASET_ZIP.exists():
        raise FileNotFoundError(f"Zip file not found at {DATASET_ZIP}")

    print("Extracting dataset...")
    with ZipFile(DATASET_ZIP, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print("Dataset extracted.")


def stemming(content: str) -> str:
    stem_content = re.sub("[^a-zA-Z]", " ", content)
    stem_content = stem_content.lower()
    stem_content = stem_content.split()
    stem_content = [port_stem.stem(word) for word in stem_content if word not in STOPWORDS]
    return " ".join(stem_content)


def main():
    extract_dataset()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Make sure sentiment140.zip contains "
            "training.1600000.processed.noemoticon.csv"
        )

    print("Loading dataset...")
    column_names = ["target", "id", "date", "flag", "user", "text"]
    twitter_data = pd.read_csv(DATA_PATH, names=column_names, encoding="ISO-8859-1")

    # target: 0 = negative, 4 = positive -> map 4 to 1
    twitter_data.replace({"target": {4: 1}}, inplace=True)

    print("Cleaning + stemming text (this takes a while on 1.6M rows)...")
    twitter_data["stem_content"] = twitter_data["text"].apply(stemming)

    X = twitter_data["stem_content"].to_numpy()
    Y = twitter_data["target"].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2, stratify=Y
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training LogisticRegression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, Y_train)

    train_acc = accuracy_score(Y_train, model.predict(X_train_vec))
    test_acc = accuracy_score(Y_test, model.predict(X_test_vec))
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy:  {test_acc:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)

    with open(MODELS_DIR / "trained_model.sav", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "vectorizer.sav", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved trained_model.sav and vectorizer.sav to {MODELS_DIR}")


if __name__ == "__main__":
    main()