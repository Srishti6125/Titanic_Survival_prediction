# 🛳️ Titanic — Survival Prediction

A machine learning notebook that analyzes the Titanic dataset to predict passenger survival using EDA, feature engineering, and an XGBoost-based classification workflow.

---

## 📁 Overview

This notebook performs an end-to-end analysis on the Titanic dataset (data cleaning → EDA → feature engineering → modeling → evaluation) to understand survival drivers and build a reliable classifier.

---

## 🧠 Objective

Predict whether a passenger survived the Titanic disaster using passenger details (age, sex, class, fare, family features, etc.).

---

## 📊 Dataset

- ✅ **Source**: [](https://www.kaggle.com/datasets/yasserh/titanic-dataset.)
-💡 Target: Survived
-⚖️ Typical columns used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Plotly
- Scikit-learn
- Logistic Regression
- Random Forest Classifier
- XGBoost

---

## 🚀 Models Trained

| Model              | Accuracy| Precision| Recall  | F1 score | Roc_auc|
|-------------------|---------|---------|----------|---------|----------|
| Logistic Regression  | 77.6%  | 69.8%  | 73.9%   | 71.8%   | 8.48% |
| Random Forest | 83.2%  | 77.4%  | 79.7%  | 78.5%   | 88.3%|
| **XGBoost**        | **84.3%** | **78.0%** | **82.6%** |**80.2%** | **86.9%** |

🎯 **Final Model Chosen**: `XGBoost Classifier`  
Due to its high accuracy, ability to handle non-linear data, and better generalization.

---

## 🔍 Feature Importance (XGBoost)

Top features impacting trip duration:
- `Sex_female`
- `Age`
- `Pclass`
- `Fare`

Used `SHAP` method to visualize gain-based feature importance.

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion matrix 
- Cross-validation for robust estimates

---

## 📌 Key Learnings

- Clean real-world data: handling missing ages, embarked, and cabin info
- Use EDA + visualization to generate hypotheses
- Create compact features that capture family structure and social cues
- Compare models and interpret feature importance for explainability

---

## ✅ Future Improvement

- Deploy with **Streamlit or Flask**
- Build a real-time prediction dashboard
- Experiment with stacking or simple ensembles

---
