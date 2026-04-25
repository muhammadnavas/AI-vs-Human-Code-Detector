# AI vs Human Code Detector

Machine learning project to classify source code as either AI-generated or human-written.
It analyzes language patterns and stylistic signals across Python, Java, and JavaScript code.

The project includes:
- A training pipeline for multiple classifiers
- Saved model artifacts under `model/`
- A Streamlit web app for interactive prediction
- Dataset folders for Python, Java, and JavaScript samples

## Features

- Binary classification: `ai` vs `human`
- Multiple models trained and evaluated:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Character-level TF-IDF feature extraction
- Ensemble voting in the app for final prediction
- Confidence display per model and overall result

## Project Structure

```text
MLProject/
  app.py
  train.py
  requirements.txt
  README.md
  data/
    python/
      ai/
      human/
    java/
      AI/
      Human/
    js/
      AI/
      Human/
  model/
    logistic.pkl
    randomforest.pkl
    gradientboost.pkl
    xgboost.pkl
    vectorizer.pkl
    labelencoder.pkl
    python/
    java/
    js/
  tableau_data/
```

## Prerequisites

- Python 3.9+
- pip

## Installation

1. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows (Command Prompt):

```bash
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

## Train Models

Run:

```bash
python train.py
```

This will:
- Load code samples from `data/`
- Build language-specific TF-IDF features
- Train models
- Print classification reports
- Save artifacts into `model/` (and language subfolders)

## Run the App

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How to Use

1. Paste code into the text area.
2. Click **Analyze Code**.
3. Review:
- Overall prediction (`AI-Generated` or `Human-Written`)
- Ensemble confidence
- Per-model predictions and confidence values

## Dataset Format

Each language folder should contain two classes:
- AI-generated code files
- Human-written code files

Expected extensions:
- Python: `.py`
- Java: `.java`
- JavaScript: `.js`

## Notes

- Prediction quality depends on dataset quality, size, and balance.
- Results are probabilistic and should be treated as supportive signals, not absolute truth.
- If model files are missing, run `python train.py` before launching the app.

## Common Issues

- `ModuleNotFoundError`: install dependencies with `pip install -r requirements.txt`.
- `Model not found` warning in app: ensure training completed and artifacts exist in `model/`.
- Streamlit command not found: use `python -m streamlit run app.py`.

## Future Improvements

- Add more languages and larger datasets
- Improve feature engineering and model calibration
- Add explainability signals for why a prediction was made
- Export prediction logs and metrics dashboard
