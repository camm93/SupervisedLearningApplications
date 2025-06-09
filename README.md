# 🤖 Supervised Machine Learning Projects

**Author**: Cristian Murillo 

A collection of supervised machine learning projects applying classification and regression techniques across diverse real-world datasets.

This repository serves as a practical record of experimentation with traditional ML pipelines using scikit-learn. Each project is self-contained and includes data preprocessing, model building, evaluation, and interpretation.

---

## 🧠 Techniques and Concepts Applied

- Classical supervised ML algorithms:
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Random Forest**
  - **Gradient Boosting**
- Dimensionality reduction with **Principal Component Analysis (PCA)**
- Model selection with **GridSearchCV**
- Cross-validation and metrics (accuracy, precision, recall, F1-score)
- Handling class imbalance
- Confusion matrix visualization
- OneHotEncoding, scaling, and custom preprocessing pipelines

---

## 📁 Project Overview

### 1. 🧪 Wine Quality Classification

- **Dataset**: UCI Wine Quality Dataset
- **Goal**: Predict wine quality (low, medium, high) based on physicochemical attributes
- **Techniques**:
  - Multiclass classification
  - Stratified train-test split
  - Logistic Regression, DecisionTreeClassificer, Random Forest Classifier, XGBClassifier, MLPClassifier comparison.
  - RandomizedSearchCV for hyperparameter tuning
  - Class imbalance handling

---

### 2. 🤟 Sign Language MNIST Classification

- **Dataset**: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Goal**: Classify hand signs (A–Y) from 28x28 grayscale images
- **Techniques**:
  - **PCA** for dimensionality reduction
  - **KNN Classifier** as final model
  - Model selection via **GridSearchCV**
  - Confusion matrix to evaluate per-class performance
- **Notes**: PCA reduces dimensionality but may impact image fidelity and model accuracy

---

### 3. 🧠 Stroke Prediction (Classification)

- **Dataset**: [Cerebral Stroke Dataset](https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset)
- **Goal**: Predict whether a patient is likely to experience a stroke
- **Techniques**:
  - Preprocessing pipeline with imputation, scaling, and encoding
  - Handling class imbalance
  - Evaluation using confusion matrix and precision-recall tradeoff

---

### 4. 🌍 Sustainable Development Goal (SDG) Text Classification

- **Dataset**: Custom curated text data labeled with UN’s 17 SDGs
- **Goal**: Classify documents into one or more SDGs based on text content
- **Techniques**:
  - **Natural Language Processing (NLP)** with scikit-learn
  - Feature engineering using **TF-IDF Vectorization**
  - Multi-class classification with Logistic Regression, Random Forest, LGBM
  - Evaluation with classification report, confusion matrix

---

## 🧪 Environment and Dependencies

All projects use Python 3.x and common ML libraries like:

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
```

# 📬 Contact
Feel free to connect with me for feedback or collaboration:
Cristian Murillo — [LinkedIn](https://www.linkedin.com/in/cristianmurillom/) | [GitHub](https://github.com/camm93)
