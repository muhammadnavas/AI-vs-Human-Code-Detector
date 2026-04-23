import streamlit as st
import joblib
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelResult:
    """Data class for storing model prediction results"""
    name: str
    prediction: str
    confidence: float

class CodeAnalyzer:
    """Main class for analyzing code with multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load all required models and encoders"""
        try:
            model_files = {
                'logistic': 'model/logistic.pkl',
                'random_forest': 'model/randomforest.pkl',
                'gradient_boost': 'model/gradientboost.pkl',
                'xgboost': 'model/xgboost.pkl'
            }
            
            for name, path in model_files.items():
                if Path(path).exists():
                    self.models[name] = joblib.load(path)
                else:
                    st.warning(f"Model {name} not found at {path}")
            
            if Path('model/vectorizer.pkl').exists():
                self.vectorizer = joblib.load('model/vectorizer.pkl')
            else:
                st.error("Vectorizer not found!")
                return
            
            if Path('model/labelencoder.pkl').exists():
                self.label_encoder = joblib.load('model/labelencoder.pkl')
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
    
    def predict_with_model(self, X: np.ndarray, model_name: str) -> Optional[ModelResult]:
        if model_name not in self.models:
            return None
            
        try:
            model = self.models[model_name]
            
            if model_name == 'xgboost':
                y_pred = model.predict(X)
                if self.label_encoder:
                    prediction = self.label_encoder.inverse_transform(y_pred)[0]
                else:
                    prediction = "ai" if y_pred[0] == 0 else "human"
            else:
                prediction = model.predict(X)[0]
            
            confidence = model.predict_proba(X).max()
            
            return ModelResult(
                name=model_name.replace('_', ' ').title(),
                prediction=prediction,
                confidence=confidence
            )
            
        except Exception as e:
            st.warning(f"Error with {model_name}: {str(e)}")
            return None
    
    def analyze_code(self, code: str) -> Tuple[List[ModelResult], str, float]:
        """Analyze code with all available models"""
        if not self.vectorizer:
            return [], "Error", 0.0
        
        X = self.vectorizer.transform([code])
        results = []
        
        for model_name in self.models.keys():
            result = self.predict_with_model(X, model_name)
            if result:
                results.append(result)
        
        final_prediction, final_confidence = self.ensemble_vote(results)
        
        return results, final_prediction, final_confidence
    
    def ensemble_vote(self, results: List[ModelResult]) -> Tuple[str, float]:
        """Calculate ensemble prediction using weighted voting"""
        if not results:
            return "unknown", 0.0
        
        ai_votes = [r.confidence for r in results if r.prediction == "ai"]
        human_votes = [r.confidence for r in results if r.prediction == "human"]
        
        ai_score = np.mean(ai_votes) * len(ai_votes) if ai_votes else 0
        human_score = np.mean(human_votes) * len(human_votes) if human_votes else 0
        
        if ai_score > human_score:
            return "ai", ai_score / len(results)
        else:
            return "human", human_score / len(results)

def render_single_code_analysis():
    st.markdown("## 📝 Analyze Single Code")
    
    if 'analyzer' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.analyzer = CodeAnalyzer()
    
    analyzer = st.session_state.analyzer

    code_input = st.text_area(
        "📝 Enter your Python code here:",
        height=300,
        placeholder="# Paste your Python code here...\nprint('Hello, World!')"
    )
    show_confidence = True
    
    if st.button("🔍 Analyze Code", type="primary"):
        if not code_input.strip():
            st.warning("⚠️ Please enter some code to analyze.")
            return
        
        with st.spinner("Analyzing code..."):
            results, final_pred, final_conf = analyzer.analyze_code(code_input)
            
            if not results:
                st.error("❌ Analysis failed. Please check if models are loaded correctly.")
                return
        
        display_single_analysis_results(results, final_pred, final_conf, show_confidence)

def display_single_analysis_results(results, final_pred, final_conf, show_confidence):
    st.markdown("## 📊 Analysis Results")

    ai_count = sum(1 for r in results if r.prediction == "ai")
    avg_conf = np.mean([r.confidence for r in results])
    status_label = "AI-Generated" if final_pred == "ai" else "Human-Written"
    status_icon = "🤖" if final_pred == "ai" else "👨‍💻"
    status_class = "status-ai" if final_pred == "ai" else "status-human"

    st.markdown(
        f"""
        <style>
        .results-grid {{
            display: grid;
            grid-template-columns: 1.1fr 1fr 1fr;
            gap: 1rem;
            align-items: stretch;
            margin-bottom: 0.75rem;
        }}
        .status-card {{
            padding: 0.85rem 1rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background: #f8fafc;
        }}
        .status-card .label {{
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
            color: #334155;
        }}
        .status-card .value {{
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.1;
        }}
        .status-ai {{
            border-left: 5px solid #ef4444;
            background: #fff1f2;
        }}
        .status-human {{
            border-left: 5px solid #22c55e;
            background: #f0fdf4;
        }}
        .metric-card {{
            padding: 0.85rem 1rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background: white;
        }}
        .metric-card .label {{
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
            color: #334155;
        }}
        .metric-card .value {{
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.1;
        }}
        </style>
        <div class="results-grid">
            <div class="status-card {status_class}">
                <div class="label">Overall Prediction</div>
                <div class="value">{status_icon} {status_label}</div>
                <div class="label" style="margin-top:0.5rem;">Confidence</div>
                <div class="value">{final_conf:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Models Voting AI</div>
                <div class="value">{ai_count}/{len(results)}</div>
            </div>
            <div class="metric-card">
                <div class="label">Average Confidence</div>
                <div class="value">{avg_conf:.1%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<hr style='margin:0.25rem 0;'>", unsafe_allow_html=True)
    render_model_confidence_graph(results)
    st.markdown("<hr style='margin:0.15rem 0 0.1rem;'>", unsafe_allow_html=True)
    st.markdown("### 🔬 Individual Model Predictions")
    
    for result in results:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{result.name}**")
        
        with col2:
            if result.prediction == "ai":
                st.write("🤖 AI-Generated")
            else:
                st.write("👨‍💻 Human-Written")
        
        with col3:
            if show_confidence:
                st.write(f"{result.confidence:.1%}")

def render_model_confidence_graph(results):
    """Render a vertical bar graph for all model confidences using Plotly."""
    import plotly.graph_objects as go
    
    st.markdown("### 📈 Model Confidence Graph")

    if not results:
        st.info("No model results available to plot.")
        return

    model_names = [r.name for r in results]
    confidences = [r.confidence * 100 for r in results]
    colors = ["#ef4444" if r.prediction == "ai" else "#22c55e" for r in results]
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=model_names,
                y=confidences,
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{c:.1f}%" for c in confidences],
                textposition="outside",
                textfont=dict(size=12, color="#1e293b"),
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>",
                showlegend=False,
            )
        ]
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(
            range=[0, 115],
            dtick=20,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(200, 200, 210, 0.2)",
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
        ),
        height=360,
        margin=dict(l=60, r=30, t=30, b=100),
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        paper_bgcolor="white",
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

def main():
    st.set_page_config(
        page_title="AI vs Human Code Detector",
        page_icon="🤖",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1320px;
            padding-top: 0.8rem;
            padding-bottom: 0.8rem;
        }
        div[data-testid="stButton"] > button {
            padding: 0.3rem 0.9rem;
            font-size: 0.9rem;
            border-radius: 0.7rem;
        }
        div[data-testid="stTextArea"] textarea {
            border-radius: 0.85rem;
        }
        h1, h2, h3 {
            margin-top: 0.1rem;
            margin-bottom: 0.25rem;
        }
        hr {
            margin: 0.35rem 0 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("AI vs Human Code Detector")
    st.markdown("**Advanced ensemble ML analysis**")

    render_single_code_analysis()
    
    st.markdown("<hr style='margin:0.5rem 0 0.25rem;'>", unsafe_allow_html=True)
    st.markdown("<small style='color:#64748b; font-style:italic;'>Predictions may vary; results are approximate.</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
