# Stroke Prediction — Exploratory & Predictive Analysis

**Author:** Sebastian Silva  
**Dataset:** [Healthcare Dataset — Stroke Data (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
**Language:** Python 3 | Jupyter Notebook

## Setup & Requirements

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy
```

### Running the Notebook

1. Download the dataset from Kaggle: `healthcare-dataset-stroke-data.csv`
2. Update the file path in Cell 1 to match your local directory
3. Run cells sequentially from top to bottom

---

## Project Overview

This project investigates stroke prediction using a real-world healthcare dataset. It combines exploratory data analysis, unsupervised clustering, and supervised classification to understand the factors associated with stroke risk and evaluate model performance on an imbalanced target variable.

---

## Dataset

The dataset contains patient-level health and demographic information with the following features:

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Patient age |
| `gender` | Categorical | Male / Female / Other |
| `hypertension` | Binary | 1 if patient has hypertension |
| `heart_disease` | Binary | 1 if patient has heart disease |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Type of employment |
| `Residence_type` | Categorical | Urban or Rural |
| `avg_glucose_level` | Numeric | Average blood glucose level |
| `bmi` | Numeric | Body mass index |
| `smoking_status` | Categorical | Smoking history |
| `stroke` | Binary (target) | 1 if patient had a stroke |

**Class imbalance:** Stroke cases represent approximately 5% of observations, making this a challenging imbalanced classification problem.

---

## Notebook Structure

### 1. Data Exploration & Preprocessing
- Loaded and inspected the dataset; dropped the non-informative `id` column
- Computed summary statistics (mean, median, standard deviation)
- Identified 201 missing values in `bmi`; applied **median imputation** to preserve all records given the medical context

### 2. Exploratory Data Analysis (EDA)
- Pairplot with stroke as hue for overall visual separation
- Count plots for stroke by gender and smoking status
- Scatter plots with regression lines for `age` vs. `bmi` and `age` vs. `avg_glucose_level` (with R² annotations)
- Violin plots and box plots comparing age, glucose, and BMI distributions by stroke outcome
- Pearson correlation analysis between numeric features and stroke outcome
- Correlation heatmap for numerical features

Key finding: `age` showed the strongest correlation with stroke among the numeric predictors. Most features show limited linear separation between classes, suggesting complex interaction effects.

### 3. Unsupervised Clustering
Clustering was applied as an exploratory step to detect natural patient groupings after preprocessing (StandardScaler + OneHotEncoder). Four algorithms were compared:

| Method | Silhouette Score |
|---|---|
| Agglomerative Clustering | 0.389 |
| K-Means (k=3) | 0.193 |
| MiniBatch K-Means (k=3) | 0.196 |
| Mean Shift (bandwidth=3) | 0.385 |

PCA (3 components) was used to visualize cluster structure in 3D. Agglomerative and Mean Shift methods produced more clearly separated clusters, consistent with their higher silhouette scores.

### 4. PCA Feature Analysis
- PCA reduced the preprocessed feature space to 3 principal components
- Component loadings were extracted and top contributing features identified for PC1, PC2, and PC3
- 3D cluster visualizations generated for all four clustering methods using PCA-reduced coordinates

<Figure size 1400x1000 with 4 Axes><img width="1107" height="990" alt="image" src="https://github.com/user-attachments/assets/232584d3-5648-4175-baa7-deadb54ff296" />


### 5. Supervised Classification (Pipelines)
Three classification models were built using `imblearn` pipelines combining:
- **Preprocessing:** `StandardScaler` for numeric features + `OneHotEncoder` for categorical features
- **Oversampling:** SMOTE (`sampling_strategy=0.10` to raise minority class proportion)
- **Classifier**

Models evaluated:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline; best recall for stroke detection |
| Random Forest | 200 estimators; highest accuracy but very low recall |
| Neural Network (MLP) | Hidden layers: (32, 16); intermediate recall |

Metrics reported: Accuracy, Precision, Recall, F1-Score, ROC-AUC. ROC curves plotted for all three models.

<Figure size 800x500 with 1 Axes><img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/8ecbfccf-ff9b-4ff5-b611-46494f3ae8c0" />


Key finding: Random Forest achieved the highest accuracy (0.932) but largely predicted the majority class, yielding very low recall (~0.10). Logistic Regression provided the best balance for identifying actual stroke cases despite lower overall accuracy.


<Figure size 800x600 with 1 Axes><img width="691" height="545" alt="image" src="https://github.com/user-attachments/assets/4c6072c0-cdf6-4d3a-8a2a-85a3f5bfbd66" />

---

## Key Takeaways

- **Class imbalance** is the central challenge; SMOTE improves minority class detection but can hurt calibration
- **Age** is the most predictive numeric feature; glucose level and BMI also contribute
- **Logistic Regression** performs best for recall on stroke cases — the clinically relevant metric
- **Clustering** reveals meaningful patient subgroups but does not map cleanly to stroke/no-stroke labels, confirming the multi-factorial nature of stroke risk
- A **pipeline approach** ensures consistent preprocessing across training and test sets and prevents data leakage
