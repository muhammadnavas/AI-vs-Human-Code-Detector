import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

LANGUAGES = {
    'python': '.py',
    'java': '.java',
    'js': '.js'  
}

LANG_CONFIGS = {
    'python': {'ngram_range': (3, 5), 'max_features': 5000, 'analyzer': 'char'},
    'java': {'ngram_range': (3, 7), 'max_features': 5000, 'analyzer': 'char'},
    'js': {'ngram_range': (2, 6), 'max_features': 5000, 'analyzer': 'char'}
}

def load_code_files(ai_folder, human_folder, extension):
    language_data = []
    
    if os.path.exists(ai_folder):
        for filename in os.listdir(ai_folder):
            if filename.endswith(extension):
                try:
                    with open(os.path.join(ai_folder, filename), "r", encoding="utf-8", errors="ignore") as f:
                        language_data.append({"code": f.read(), "label": "ai"})
                except: continue
    
    if os.path.exists(human_folder):
        for filename in os.listdir(human_folder):
            if filename.endswith(extension):
                try:
                    with open(os.path.join(human_folder, filename), "r", encoding="utf-8", errors="ignore") as f:
                        language_data.append({"code": f.read(), "label": "human"})
                except: continue
    return language_data

def generate_data():
    os.makedirs("tableau_data", exist_ok=True)
    
    all_metrics = []
    all_cm = []
    all_stats = []
    all_importance = []
    
    for lang, ext in LANGUAGES.items():
        print(f"Processing {lang}...")
        
        # Load Data
        ai_folder = f"data/{lang}/AI" if os.path.exists(f"data/{lang}/AI") else f"data/{lang}/ai"
        human_folder = f"data/{lang}/Human" if os.path.exists(f"data/{lang}/Human") else f"data/{lang}/human"
        
        lang_data = load_code_files(ai_folder, human_folder, ext)
        if not lang_data:
            print(f"No data for {lang}")
            continue
            
        df = pd.DataFrame(lang_data)
        
        # Dataset Stats
        stats = df['label'].value_counts().reset_index()
        stats.columns = ['Label', 'Count']
        stats['Language'] = lang
        all_stats.append(stats)
        
        # Vectorizer and Models
        model_dir = f"model/{lang}"
        if not os.path.exists(model_dir):
            print(f"No models for {lang}")
            continue
            
        vectorizer = joblib.load(f"{model_dir}/vectorizer.pkl")
        le = joblib.load(f"{model_dir}/label_encoder.pkl")
        
        X = vectorizer.transform(df['code'])
        y = df['label']
        
        # Split (same random state as train.py)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_test_enc = le.transform(y_test)
        
        model_names = ['logistic', 'random_forest', 'gradient_boost', 'xgboost']
        
        for m_name in model_names:
            model_path = f"{model_dir}/{m_name}.pkl"
            if not os.path.exists(model_path):
                continue
                
            model = joblib.load(model_path)
            
            # Predictions
            if m_name == 'xgboost':
                y_pred_enc = model.predict(X_test)
                y_pred = le.inverse_transform(y_pred_enc)
            else:
                y_pred = model.predict(X_test)
                y_pred_enc = le.transform(y_pred)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=['ai', 'human'])
            
            all_metrics.append({
                'Language': lang,
                'Model': m_name,
                'Accuracy': acc,
                'AI_Precision': precision[0],
                'AI_Recall': recall[0],
                'AI_F1': f1[0],
                'Human_Precision': precision[1],
                'Human_Recall': recall[1],
                'Human_F1': f1[1]
            })
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=['ai', 'human'])
            for i, true_label in enumerate(['ai', 'human']):
                for j, pred_label in enumerate(['ai', 'human']):
                    all_cm.append({
                        'Language': lang,
                        'Model': m_name,
                        'TrueLabel': true_label,
                        'PredictedLabel': pred_label,
                        'Count': cm[i, j]
                    })
            
            # Feature Importance (if applicable)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top 20 features
                indices = np.argsort(importances)[::-1][:20]
                for idx in indices:
                    all_importance.append({
                        'Language': lang,
                        'Model': m_name,
                        'Feature': feature_names[idx],
                        'Importance': importances[idx]
                    })

    # Save to CSV
    pd.DataFrame(all_metrics).to_csv("tableau_data/model_performance.csv", index=False)
    pd.concat(all_stats).to_csv("tableau_data/dataset_distribution.csv", index=False)
    pd.DataFrame(all_cm).to_csv("tableau_data/confusion_matrices.csv", index=False)
    pd.DataFrame(all_importance).to_csv("tableau_data/feature_importance.csv", index=False)
    
    print("Tableau datasets generated in 'tableau_data/' folder.")

if __name__ == "__main__":
    generate_data()
