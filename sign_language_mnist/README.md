# Sign Language MNIST Classification - PCA + KNN

**Author**: Cristian Murillo  
**Supervised Machine Learning Project**

---

## 📌 Problem Description

Automatic recognition of sign language has important applications in accessibility, especially for people with hearing impairments. This project uses a classical machine learning approach to classify images of the American Sign Language (ASL) alphabet using the Kaggle dataset: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data).

The goal is to build a classifier that recognizes individual letters represented by hand gestures, using a pipeline consisting of dimensionality reduction with PCA followed by classification using K-Nearest Neighbors (KNN).

---

## 📂 Dataset

The dataset contains grayscale images of 28x28 pixels representing letters from the ASL alphabet (excluding J and Z, as they require motion). Each image is flattened into a vector of 784 features (28 × 28).

- Total samples:  
  - **Train**: 27,455 images  
  - **Test**: 7,172 images  

- Labels: Letters A–Y (excluding J and Z)

---

## ⚙️ Project Workflow

1. **Data Loading and Preprocessing**
    - Load the CSV into a DataFrame
    - Normalize pixel values to range [0, 1]

2. **Dimensionality Reduction with PCA**
    - Apply PCA to reduce the dimensionality of image vectors
    - Retain 95% of the variance
    - Note: PCA introduces some visual degradation due to loss of pixel-level detail

3. **Classification with KNN**
    - Use `KNeighborsClassifier` as the base model
    - Hyperparameter tuning via `GridSearchCV`
    - Stratified cross-validation to ensure balanced class distribution

4. **Model Evaluation**
    - Compute confusion matrix to analyze performance per class
    - Report metrics like Accuracy and F1-score
    - Compare performance with and without PCA

---

## 🔎 Model Selection

The best hyperparameters are selected using `GridSearchCV` over a pipeline consisting of PCA + KNN. The grid search explores various values for:

- Number of PCA components
- Number of neighbors (k) in KNN
- Weights (`uniform`, `distance`)

```python
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
```

## 📊 Results
* **Test set accuracy**: ~0.82 (with PCA)
* **Observation**: While PCA reduces dimensionality and speeds up training, it also causes information loss in image data, slightly hurting model performance.
* **Confusion Matrix**:
    * Good overall performance
    * Some confusion between visually similar letters (e.g., N vs E and O vs. C)

## 📌 Conclusions and Recommendations
* PCA significantly reduces data dimensionality, improving computational efficiency, but at the cost of information loss, which can be problematic for image-based tasks.
* KNN is an effective classifier for this problem, but may not scale well to large datasets unless combined with dimensionality reduction techniques.

### Future improvements could include:
* Applying data augmentation to increase diversity
* Comparing other classifiers (SVM, Random Forest) to KNN
