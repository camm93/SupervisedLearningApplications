# 📌 Project 1: Classifying Texts According to the UN Sustainable Development Goals (SDGs)

## 🌍 Context
In 2015, the United Nations adopted the 2030 Agenda for Sustainable Development, which sets 17 Sustainable Development Goals (SDGs) and 169 targets aimed at reducing poverty, improving health and education, achieving gender equality, and minimizing environmental impact.

A key challenge is analyzing and interpreting citizen-participation texts to align them with these SDGs. Manual classification is labor-intensive and requires domain expertise.

## 🎯 Objective
Develop a machine learning pipeline using NLP techniques to automatically classify Spanish-language texts according to the 17 SDGs. The goal is to create a scalable tool that can support decision-making in public policy by highlighting which goals are most represented in citizen feedback.

---

# 🧱 Project Structure

## 1. Initial Data Processing
- Loading and cleaning raw text data.
- Removing duplicates.

## 2. Exploratory Data Analysis (EDA)
- Analysis of class distribution (heavily imbalanced across SDGs).
- Visualizations such as word clouds and document frequency per class.

## 3. Text Preprocessing
- Lowercasing, removing punctuation and stopwords.
- Tokenization using regular expressions.
- Stemming with SnowballStemmer (Spanish).
- Removing accents with `unidecode`.

## 4. Feature Engineering
- Vectorization using TF-IDF (top 5000 terms).
- Dimensionality reduction using:
  - TruncatedSVD (for model input),
  - t-SNE (for 2D semantic visualization).

## 5. Modeling Approaches

Two modeling phases were implemented:

### A. Without Class Balancing
- Models: Logistic Regression, Random Forest, LightGBM.
- Evaluation: F1 macro-score.
- Hyperparameter tuning with RandomizedSearchCV.
- Cross-validation: Stratified K-Fold.
- Metrics: Accuracy, F1-score, confusion matrix.

**Best model:** Logistic Regression  
**Optimal Hyperparameters:**  
```python
{'tfidf__ngram_range': (1, 2),
 'tfidf__max_features': None,
 'svd__n_components': 100,
 'model__solver': 'liblinear',
 'model__penalty': 'l1',
 'model__C': 10}
```

### B. With SMOTE Oversampling
- Used SMOTE to generate synthetic samples for underrepresented SDG classes.
- Repeated the same model comparison and hyperparameter tuning.

**Best model again**: Logistic Regression with identical optimal parameters.
---

## 🔍 Results & Insights
- Surprisingly, Logistic Regression consistently outperformed more complex models like Random Forest and LightGBM.
- Class balancing with SMOTE had **minimal impact** on overall performance.
- Overall accuracy: **0.86** (both with and without SMOTE).
- High F1-scores (>90) for well-represented goals (e.g., SDG 3, 4, 5).
- Low F1-scores (<70) for goals with limited data (e.g., SDG 8, 9, 10).


## ✅ Conclusions
- The model is capable of identifying SDG-relevant topics from Spanish texts with good overall accuracy.
- Logistic Regression remains a strong baseline, particularly when paired with proper feature extraction and regularization.
- There is a direct correlation between class representation and model performance.

## 💡 Recommendations for Future Work
- Collect more labeled examples, especially for underrepresented SDGs.
- Test more advanced models such as neural networks and transformers (e.g., BERT multilingual).
- Explore contextual embeddings (e.g., using spaCy, SBERT).
- Expand hyperparameter search space and experiment with different feature selection techniques.

## 🔧 Tech Stack
| Category            | Libraries Used                                                                        |
| ------------------- | ------------------------------------------------------------------------------------- |
| Text Preprocessing  | `nltk`, `unidecode`, `SnowballStemmer`, `RegexpTokenizer`                             |
| Visualization       | `matplotlib`, `wordcloud`, `TSNE`                                                     |
| ML Models           | `LogisticRegression`, `RandomForestClassifier`, `LGBMClassifier`, `XGBClassifier`     |
| Feature Engineering | `TfidfVectorizer`, `TruncatedSVD`                                                     |
| Evaluation & Tuning | `f1_score`, `classification_report`, `RandomizedSearchCV`, `StratifiedKFold`, `SMOTE` |
| Pipeline Management | `Pipeline` from `sklearn`                                                             |

## Contributor
Diego Pedreros
