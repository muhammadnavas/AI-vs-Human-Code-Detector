# Tableau Visualization Datasets

This directory contains CSV files generated for visualizing the performance and data distribution of the AI-vs-Human Code Detector project.

## Files Overview:

1. **`model_performance.csv`**
   - **Columns**: `Language`, `Model`, `Accuracy`, `AI_Precision`, `AI_Recall`, `AI_F1`, `Human_Precision`, `Human_Recall`, `Human_F1`.
   - **Usage**: Use this to compare model performance across different programming languages. Bar charts or radar charts are ideal here.

2. **`dataset_distribution.csv`**
   - **Columns**: `Label`, `Count`, `Language`.
   - **Usage**: Shows the balance of your training/testing data. Use pie charts or stacked bar charts to visualize the distribution of AI vs Human samples per language.

3. **`confusion_matrices.csv`**
   - **Columns**: `Language`, `Model`, `TrueLabel`, `PredictedLabel`, `Count`.
   - **Usage**: Create heatmaps to see where models are making mistakes (e.g., Human code being flagged as AI).

4. **`feature_importance.csv`**
   - **Columns**: `Language`, `Model`, `Feature`, `Importance`.
   - **Usage**: Shows the top 20 character n-grams that the models (Random Forest, Gradient Boosting, XGBoost) find most significant. Use bar charts or word clouds.

## Recommended Tableau Visualizations:

- **Model Comparison Heatmap**: Language on one axis, Model on another, and Accuracy as the color intensity.
- **Precision-Recall Scatter Plot**: AI_Precision vs AI_Recall to see which models are most "careful" vs "sensitive" to AI code.
- **Feature Word Cloud**: Use `Feature` as text and `Importance` as size to see what "style" of code indicates AI generation.
