<<<<<<< HEAD
# Twitter Sentiment Analysis — Streamlit App (VS Code project)

Deploys the Logistic Regression + TF-IDF model from your notebook as an interactive web app.

## Folder structure

```
twitter-sentiment-analysis/
├── .vscode/
│   ├── launch.json      # Run/Debug configs (Streamlit + training script)
│   └── settings.json    # Points VS Code at the venv interpreter
├── data/                 # Put the Kaggle CSV here (gitignored)
├── models/                # trained_model.sav + vectorizer.sav land here (gitignored)
├── app.py                 # Streamlit app
├── train_model.py         # Training script
├── requirements.txt
└── .gitignore
```

## What was fixed from the notebook

Your notebook only saved `trained_model.sav` (the LogisticRegression model) — it never saved
the fitted `TfidfVectorizer`. Without that *exact* fitted vectorizer, new text can't be turned
into the same feature space the model was trained on. `train_model.py` now saves both
`models/trained_model.sav` **and** `models/vectorizer.sav`.

## Setup in VS Code

1. **Open the folder** in VS Code: `File → Open Folder → twitter-sentiment-analysis`

2. **Create a virtual environment** (Terminal → New Terminal):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   VS Code should auto-detect `venv` as the interpreter (bottom-right corner). If not, run
   **Python: Select Interpreter** from the Command Palette and pick `venv`.

3. **Get the dataset**: download `sentiment140` from Kaggle
   (`kazanova/sentiment140`) and place `training.1600000.processed.noemoticon.csv` inside `data/`.

4. **Train the model** — either:
   - Run in terminal: `python train_model.py`, or
   - Press **F5** and pick **"Python: train_model.py"**

   This writes `models/trained_model.sav` and `models/vectorizer.sav`.

5. **Run the app** — either:
   - Terminal: `streamlit run app.py`, or
   - Press **F5** and pick **"Streamlit: app.py"** (runs with the debugger attached, so you
     can set breakpoints in `app.py`)

   Streamlit opens the app in your browser at `http://localhost:8501`.

## Features

- **Single Prediction tab** — type any text, get Positive/Negative with a confidence score
- **Batch Prediction tab** — upload a CSV with a `text` column, get predictions for every row,
  download the results

## Deploying it publicly

Once it runs locally, push this folder to GitHub (the `.gitignore` already excludes the large
CSV, the venv, and the `.sav` files) and deploy on **Streamlit Community Cloud** (share.streamlit.io).
Since `models/*.sav` is gitignored, either commit the model files directly (if under GitHub's
100MB limit) or use [Git LFS](https://git-lfs.github.com/) / host them on Hugging Face Hub and
download them at app startup.
=======
# Twitter-Sentiment-Analysis
>>>>>>> a02d676e1ce0acd29c557e2bee96a44af1976eb4
