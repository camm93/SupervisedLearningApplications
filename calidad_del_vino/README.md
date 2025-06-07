# 🍷 Clasificación de la Calidad del Vino - Machine Learning Supervisado

**Autor**: Cristian Murillo  
**Dataset**: Extraído de [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

---

## 📌 Descripción del Problema

La industria vinícola busca constantemente mejorar la calidad de sus productos. Un modelo de clasificación de calidad permitiría identificar qué características fisicoquímicas afectan la percepción de calidad y ayudaría a optimizar el proceso de producción.

El objetivo de este proyecto es diseñar un clasificador que prediga la calidad del vino basado en sus propiedades químicas.

---

## 📂 Dataset

El archivo utilizado es `wine quality.csv`, el cual contiene **6,497 muestras** con las siguientes variables:

- `fixed acidity`  
- `volatile acidity`  
- `citric acid`  
- `residual sugar`  
- `chlorides`  
- `free sulfur dioxide`  
- `total sulfur dioxide`  
- `density`  
- `pH`  
- `sulphates`  
- `alcohol`  

La variable objetivo es `quality`, una variable discreta en el rango [1, 10], aunque en la práctica, la mayoría de los valores están entre 3 y 8. Este es un problema de **clasificación multiclase**.

---

## 🧱 Contenido del Proyecto

1. Configuración del Ambiente
2. Análisis Exploratorio de Datos
    - Valores nulos
    - Duplicados
3. Preprocesamiento
    - Mapeo de clases
    - Remoción de la clase 9 por falta de muestras
4. Modelamiento Base
5. Selección de Modelo
    - Construcción de Pipelines
    - Reescalado de variables
    - Remuestreo (SMOTE)
    - Búsqueda de hiperparámetros
    - Validación cruzada estratificada
6. Análisis de Resultados
7. Conclusiones y Recomendaciones

---

## 🚀 Modelamiento Base

Se construyó un modelo inicial para establecer expectativas y validar la metodología. La **métrica principal de evaluación** fue el **F1 Macro**, más robusta ante desbalance de clases en comparación con la `accuracy`.

---

## 🤖 Selección de Modelo

Se evaluaron múltiples clasificadores usando pipelines completos con reescalado y, opcionalmente, remuestreo:

### 📊 Modelos Evaluados

- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `XGBClassifier`
- `MLPClassifier` (Red neuronal multicapa)

### 🔄 Componentes del Pipeline

- **Imputación**: `SimpleImputer` por seguridad en caso de valores faltantes futuros.
- **Estandarización**: `StandardScaler`.
- **Remuestreo**: con `SMOTE`, para mitigar el desbalance de clases.
- **Validación Cruzada**: `StratifiedKFold`.
- **Optimización**: `RandomizedSearchCV` (200 combinaciones).

### 📌 Hiperparámetros evaluados

- Pesos por clase (`class_weight='balanced'`)
- Parámetros de regularización (`C`, `reg_lambda`)
- Solvers (`liblinear`, `lbfgs`, `saga`)
- Funciones de activación (`relu`, `logistic`)
- Número de estimadores, profundidad máxima, tasa de aprendizaje
- Re-muestreo (`SMOTE(k_neighbors=3, 4, 5)`)

### 🏆 Mejor Modelo

```python
Best Parameters:
{
  'sampling': SMOTE(k_neighbors=3, random_state=50),
  'sampling__k_neighbors': 4,
  'model': RandomForestClassifier(random_state=50),
  'model__n_estimators': 200,
  'model__max_depth': 20,
  'model__min_samples_split': 2,
  'model__class_weight': 'balanced'
}
```

## 📈 Resultados y Observaciones
* **Random Forest** fue el modelo con mejor desempeño, seguido por `XGBoost, MLPClassifier, DecisionTree, y por último LogisticRegression`.
* El modelo tuvo mejor desempeño en las clases 5, 6 y 7 — las más representadas en el conjunto de datos.
* La clase 9 fue eliminada debido a solo contar con 5 instancias, lo que hace inviable un aprendizaje significativo incluso con SMOTE.
* Se observaron aún altos niveles de falsos positivos y negativos, lo que indica margen de mejora.

---

## ✅ Conclusiones y Recomendaciones
* El desbalance de clases afectó significativamente el rendimiento de los modelos.
* La métrica `f1_macro` fue adecuada para evaluar de forma justa el desempeño general.
* Para mejorar el rendimiento:
    * Recolectar más datos, especialmente para clases poco representadas.
    * Explorar técnicas de cost-sensitive learning o focal loss.
    * Probar modelos más complejos o específicos como CatBoost o TabNet.
    * Considerar técnicas de ensemble para combinar lo mejor de cada modelo.
* Seleccionar la métrica de evaluación basada en el objetivo de negocio (ej. priorizar recall vs. precision en casos de control de calidad).
---

## 🔧 Librerías Utilizadas
| Categoría              | Librerías                                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- |
| Procesamiento de datos | `pandas`, `numpy`, `matplotlib`, `seaborn`                                                                 |
| Preprocesamiento       | `SimpleImputer`, `StandardScaler`, `SMOTE`, `Pipeline`                                                     |
| Modelos                | `LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`, `XGBClassifier`, `MLPClassifier` |
| Evaluación             | `classification_report`, `f1_score`, `confusion_matrix`, `ConfusionMatrixDisplay`                          |
| Optimización           | `RandomizedSearchCV`, `StratifiedKFold`                                                                    |
