🧠 Cerebral Stroke Prediction - Anomaly Detection with PCA

**Author**: Cristian Murillo  
**Supervised Machine Learning Project**

---

## 📌 Project Description

This project explores the use of **Principal Component Analysis (PCA)** for **unsupervised anomaly detection** on the imbalanced [Cerebral Stroke Prediction dataset](https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset) from Kaggle.

The underlying assumption is that **normal data points** will be **close to their reconstructed values** after projecting through PCA and back to the original space, while **anomalies** (such as stroke cases) will show **larger reconstruction errors**.

---

## 📂 Dataset

The dataset includes medical and lifestyle features associated with the risk of stroke, such as age, hypertension, heart disease, smoking status, etc.

- Highly **imbalanced**: Only ~2% of samples correspond to stroke cases.
- Goal: Identify stroke cases as anomalies based on reconstruction error from PCA.

---

## ⚙️ Project Workflow

### 🔧 Preprocessing

- Drop duplicate rows
- Handle missing values:
  - **Numerical features** → imputed with median
  - **Categorical features** → imputed with mode
  - `"smoking_status"` → imputed with `"unknown"` due to large number of missing values
- Scale numerical features using `StandardScaler`
- One-hot encode categorical variables using `OneHotEncoder`

### 📉 Modeling: PCA for Anomaly Detection

1. **Apply PCA**
   - Reduce dimensionality of data
   - Visualize in principal component space

2. **Reconstruct original data**
   - Calculate reconstruction error for each point
   - Set a threshold to define anomalies

3. **Evaluation**
   - Compare true labels to predicted anomalies
   - Plot confusion matrix and analyze precision, recall

---

## 📊 Key Observations

- The **mean reconstruction error** for **normal points** is **higher** than for actual stroke cases (anomalies).
- **Thresholding** the reconstruction error yields poor separation between classes.
- **Excessive false positives** lead to:
  - Positive class **precision of 0.02**
  - Accuracy ~**0.72** — misleading due to imbalance
- **Conclusion**:
  - PCA is **not suitable** for anomaly detection on this dataset
  - Either the **preprocessing pipeline** or PCA itself fails to capture relevant patterns

---

## ⚠️ Limitations and Takeaways

- PCA assumes linearity and does not model class-specific variance — not ideal for this task.
- The reconstruction error did not provide sufficient contrast to separate stroke from non-stroke cases.
- There's a clear **trade-off** between false positives and false negatives depending on the threshold.
- Consider more advanced anomaly detection techniques (e.g., Autoencoders, Isolation Forest, or One-Class SVM) or reframe as a **supervised classification** problem with class imbalance handling.
