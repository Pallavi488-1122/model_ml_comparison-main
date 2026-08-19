# Customer ML Model Comparison

## Project Overview

This project compares different Machine Learning approaches on a customer purchase dataset.

The project includes:

* Supervised Learning
* Unsupervised Learning
* Model Evaluation
* Model Comparison

The main goal is to predict whether a customer will make a purchase and identify natural customer groups using clustering.

---

## Dataset

The dataset contains **20 customer records** and **6 columns**.

### Features

* Age
* Annual_Income
* Spending_Score
* Work_Experience
* Family_Size

### Target

* Purchase: Yes / No

---

## Machine Learning Models

### 1. Logistic Regression

A supervised classification algorithm used to predict customer purchase behavior.

### 2. Decision Tree

A supervised classification algorithm that uses decision rules to predict customer purchases.

### 3. K-Means Clustering

An unsupervised learning algorithm used to identify natural groups of customers without using the Purchase target.

---

## Evaluation Metrics

The supervised models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |     1.00 |      1.00 |   1.00 |     1.00 |
| Decision Tree       |     1.00 |      1.00 |   1.00 |     1.00 |

Both supervised models achieved the same performance on the test set.

Since the dataset contains only 20 records and the test set contains only 4 records, the results should not be considered proof of perfect generalization.

---

## K-Means Analysis

K-Means clustering was applied using the customer features without using the Purchase target.

The resulting clusters were then compared with the actual Purchase values to understand whether the natural customer groups were related to purchasing behavior.

---

## Project Structure

```text
customer_ml_comparison/
│
├── data/
│   └── customer_data.csv
│
├── output/
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_decision_tree.png
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   ├── decision_tree.png
│   ├── kmeans_clusters.png
│   ├── cluster_distribution.png
│   ├── cluster_purchase_count.csv
│   └── cluster_purchase_percentage.csv
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## How to Run

### 1. Install required packages

```bash
pip install -r requirements.txt
```

### 2. Run the project

```bash
python main.py
```

The results and visualizations will be saved in the `output` folder.

---

## ML Concepts Covered

### ML-1: Fundamentals

* Data loading
* Data cleaning
* Feature selection
* Target encoding
* Feature scaling
* Train/test split

### ML-2: Supervised Learning

* Logistic Regression
* Decision Tree

### ML-3: Unsupervised Learning

* K-Means Clustering
* Customer segmentation

### ML-4: Model Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Model comparison

---

## Final Conclusion

Logistic Regression and Decision Tree achieved equal performance on the available test set.

Therefore, there is no clear performance-based winner between the two models.

For a more reliable conclusion, the models should be tested on a larger dataset with more observations.
