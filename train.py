import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

LANGUAGES = {
    'python': '.py',
    'java': '.java',
    'js': '.js'  
}

LANG_CONFIGS = {
    'python': {
        'ngram_range': (3, 5),
        'max_features': 5000,
        'analyzer': 'char'
    },
    'java': {
        'ngram_range': (3, 7),
        'max_features': 5000,
        'analyzer': 'char'
    },
    'js': {
        'ngram_range': (2, 6),
        'max_features': 5000,
        'analyzer': 'char'
    }
}

data = []

def load_code_files(ai_folder, human_folder, extension):
    language_data = []
    
    ai_path = os.path.join(ai_folder)
    if os.path.exists(ai_path):
        for filename in os.listdir(ai_path):
            if filename.endswith(extension):
                with open(os.path.join(ai_path, filename), "r", encoding="utf-8", errors="ignore") as f:
                    language_data.append({
                        "code": f.read(),
                        "label": "ai",
                        "language": [k for k, v in LANGUAGES.items() if v == extension][0]
                    })
    
    human_path = os.path.join(human_folder)
    if os.path.exists(human_path):
        for filename in os.listdir(human_path):
            if filename.endswith(extension):
                with open(os.path.join(human_path, filename), "r", encoding="utf-8", errors="ignore") as f:
                    language_data.append({
                        "code": f.read(),
                        "label": "human",
                        "language": [k for k, v in LANGUAGES.items() if v == extension][0]
                    })
    
    return language_data

for lang, ext in LANGUAGES.items():
    if os.path.exists(f"data/{lang}/AI"):
        ai_folder = f"data/{lang}/AI"
    else:
        ai_folder = f"data/{lang}/ai"
        
    if os.path.exists(f"data/{lang}/Human"):
        human_folder = f"data/{lang}/Human"
    else:
        human_folder = f"data/{lang}/human"
    data.extend(load_code_files(ai_folder, human_folder, ext))

print(f"Total samples loaded: {len(data)}")
for lang in LANGUAGES.keys():
    lang_count = sum(1 for item in data if item['language'] == lang)
    print(f"{lang.capitalize()} files: {lang_count}")

df = pd.DataFrame(data)
vectorizers = {}
for lang in LANGUAGES.keys():
            analyzer=config['analyzer'],
            ngram_range=config['ngram_range'],
            max_features=config['max_features'],
            token_pattern=r'(?u)\b\w+\b',
            strip_accents='unicode',
            lowercase=True
        )
        
        X_lang = vectorizer.fit_transform(lang_df["code"])
        vectorizers[lang] = {
            'vectorizer': vectorizer,
            'X': X_lang,
            'y': lang_df["label"]
        }

splits = {}
for lang, data in vectorizers.items():
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y'], 
        test_size=0.2, 
        random_state=42
    )
    splits[lang] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

models = {}
for lang in LANGUAGES.keys():
    if lang not in splits:
        print(f"\nSkipping {lang} - no data found")
        continue
        
    print(f"\n=== Training models for {lang.upper()} ===")
    split = splits[lang]
    X_train, X_test = split['X_train'], split['X_test']
    y_train, y_test = split['y_train'], split['y_test']
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    lang_models = {}
    
    print(f"\n📊 Training Logistic Regression for {lang}...")
    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    print(classification_report(y_test, y_pred_log))
    lang_models['logistic'] = log_model
    
    print(f"\n🌲 Training Random Forest for {lang}...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    lang_models['random_forest'] = rf_model
    
    # --- Gradient Boosting ---
    gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    print(classification_report(y_test, y_pred_gb))
    lang_models['gradient_boost'] = gb_model
    
    # --- XGBoost ---
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train_enc)
    y_pred_xgb = xgb_model.predict(X_test)
    print(classification_report(y_test_enc, y_pred_xgb, target_names=le.classes_))
    lang_models['xgboost'] = xgb_model
    
    models[lang] = {
        'vectorizer': vectorizers[lang]['vectorizer'],
        'models': lang_models,
        'label_encoder': le
    }
    

print("\n💾 Saving models and vectorizers...")
for lang, model_data in models.items():
    lang_dir = f"model/{lang}"
    os.makedirs(lang_dir, exist_ok=True)
    
    joblib.dump(model_data['vectorizer'], f"{lang_dir}/vectorizer.pkl")
    
    for model_name, model in model_data['models'].items():
        joblib.dump(model, f"{lang_dir}/{model_name}.pkl")
    
        joblib.dump(model, f"{lang_dir}/{model_name}.pkl")
    
    joblib.dump(model_data['label_encoder'], f"{lang_dir}/label_encoder.pkl\")


# Save both models + vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(log_model, "model/logistic.pkl")
joblib.dump(rf_model, "model/randomforest.pkl")
joblib.dump(gb_model, "model/gradientboost.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(xgb_model, "model/xgboost.pkl")

print("✅ Model and vectorizer saved in 'model/' folder")
