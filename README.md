# Time Series Pattern Classification

This project focuses on detecting repeating patterns in time-series data using machine learning. The goal is to distinguish between two classes by analyzing temporal behavior in short sequences.

The dataset consists of sequences with **10 time steps and 2 numerical features per timestep**, making the input shape:

X → (samples, 10, 2)
y → (samples,)


---

# Problem Objective

The objective is to:

- analyze temporal patterns in the dataset
- discover repeating patterns associated with **class 1**
- engineer meaningful features from time-series sequences
- train a machine learning model to classify sequences
- evaluate whether the model could be used in a production setting

---

# Dataset Characteristics

The dataset is **highly imbalanced**.

| Class | Percentage |
|------|------------|
| Class 0 | ~99.93% |
| Class 1 | ~0.07% |

Because of this imbalance, evaluation focuses on:

- **Recall**
- **F1-score**
- **ROC-AUC**
- **PR-AUC**

rather than accuracy.

---

# Exploratory Data Analysis

Sequences from both classes were visualized to understand temporal patterns.

Key observations:

- Class 0 sequences are relatively stable
- Class 1 sequences show a **distinct valley-shaped dip around the middle timesteps (3–6)**
- Class 1 sequences also exhibit **higher variability and stronger temporal transitions**

These observations guided the feature engineering process.

---

# Feature Engineering

Since traditional machine learning models require tabular input, time-series sequences were transformed into statistical features.

The engineered features include:

### Global statistics
- mean
- standard deviation
- minimum / maximum
- range

### Middle-window features
Capturing the pattern around timesteps **3–6**.

### Temporal difference features
Capturing changes between consecutive timesteps.

### Sequence shape features
Comparing early vs late parts of the sequence.

### Cross-feature relationships
Capturing interactions between the two signals.

---

# Model Training

Two models were evaluated:

### Logistic Regression
Used as a baseline model with feature scaling.

### Random Forest
Chosen as the final model because it captures non-linear relationships between engineered features.

To reduce randomness and overfitting, **Stratified K-Fold Cross Validation** was used during model selection.

---

# Model Evaluation

The final Random Forest model was evaluated on a test dataset.

| Metric | Result |
|------|--------|
ROC-AUC | ~0.95 |
PR-AUC | ~0.028 |
Recall | ~0.72 |
Precision | ~0.008 |

Because the dataset is extremely imbalanced, **PR-AUC and Recall are the most meaningful metrics**.

Although precision is low, the model successfully detects a large portion of the rare class 1 events.

---

# Feature Importance

Feature importance analysis from the Random Forest model shows that the classifier relies heavily on:

- variability features
- middle-window statistics
- temporal change metrics
- cross-feature relationships

This confirms that the model learned the **temporal dip pattern discovered during exploratory analysis**.

---

# Production Considerations

The model demonstrates potential for detecting rare patterns in time-series data. However, several limitations must be considered before deployment.

### Limitations

- extreme class imbalance
- low precision leading to many false positives
- reliance on patterns observed in training data

### Handling unseen patterns

If new temporal patterns appear in production data, the model may require:

- periodic retraining
- monitoring of metrics such as PR-AUC and recall
- updates to feature engineering

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# Repository Structure
time-series-pattern-classification

data/
X.npz
y.npz

notebooks/
assignment_analysis.ipynb

