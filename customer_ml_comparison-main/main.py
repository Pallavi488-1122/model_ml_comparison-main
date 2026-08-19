# ============================================================
# CUSTOMER ML MODEL COMPARISON
# ML-1 Fundamentals + ML-2 Supervised + ML-3 Unsupervised
# ML-4 Evaluation
# ============================================================

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "customer_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("CUSTOMER PURCHASE ML MODEL COMPARISON")
print("=" * 70)

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Work_Experience",
    "Family_Size",
    "Purchase"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. PURCHASE DISTRIBUTION BEFORE ENCODING
# ============================================================

print("\n" + "=" * 70)
print("PURCHASE DISTRIBUTION")
print("=" * 70)

print(df["Purchase"].value_counts())


# ============================================================
# 6. CLEAN PURCHASE COLUMN
# ============================================================

print("\n" + "=" * 70)
print("CLEANING TARGET VARIABLE")
print("=" * 70)

# Convert to string
df["Purchase"] = df["Purchase"].astype(str)

# Remove leading/trailing spaces
df["Purchase"] = df["Purchase"].str.strip()

# Convert to lowercase
df["Purchase"] = df["Purchase"].str.lower()

print("\nCleaned Purchase Values:")
print(df["Purchase"].value_counts())


# ============================================================
# 7. ENCODE PURCHASE
# ============================================================

df["Purchase"] = df["Purchase"].map({
    "no": 0,
    "yes": 1
})


# ============================================================
# 8. VALIDATE TARGET
# ============================================================

print("\nEncoded Purchase:")
print(df["Purchase"].value_counts(dropna=False))

print("\nMissing Values After Encoding:")
print(df["Purchase"].isna().sum())


# If any invalid value exists, stop execution
if df["Purchase"].isna().any():

    print("\nERROR: Invalid values found in Purchase column.")

    raise ValueError(
        "Purchase column contains values other than Yes/No. "
        "Please check customer_data.csv."
    )


# Convert target to normal integer type
df["Purchase"] = df["Purchase"].astype(int)

print("\nPurchase Data Type:")
print(df["Purchase"].dtype)

print("\nUnique Purchase Values:")
print(sorted(df["Purchase"].unique()))


# ============================================================
# 9. CHECK ALL NUMERIC FEATURES
# ============================================================

feature_columns = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Work_Experience",
    "Family_Size"
]

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 10. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print(df.isnull().sum())

if df[feature_columns].isnull().any().any():

    print(
        "\nMissing numeric values detected."
    )

    df = df.dropna(
        subset=feature_columns
    ).reset_index(drop=True)

    print(
        "Rows after removing missing values:",
        len(df)
    )


# ============================================================
# 11. DEFINE FEATURES AND TARGET
# ============================================================

X = df[feature_columns]

y = df["Purchase"].astype(int)


print("\n" + "=" * 70)
print("FEATURES AND TARGET")
print("=" * 70)

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)

print("\ny Data Type:")
print(y.dtype)

print("\nUnique Target Values:")
print(sorted(y.unique()))


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# ============================================================
# 13. FEATURE SCALING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("\nFeature scaling completed successfully.")


# ============================================================
# 14. MODEL 1 - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 - LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

y_pred_logistic = logistic_model.predict(
    X_test_scaled
)

print("\nLogistic Regression Predictions:")
print(y_pred_logistic)


# ============================================================
# 15. LOGISTIC REGRESSION METRICS
# ============================================================

logistic_accuracy = accuracy_score(
    y_test,
    y_pred_logistic
)

logistic_precision = precision_score(
    y_test,
    y_pred_logistic,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    y_pred_logistic,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    y_pred_logistic,
    zero_division=0
)

print("\nLogistic Regression Evaluation:")

print(
    f"Accuracy : {logistic_accuracy:.4f}"
)

print(
    f"Precision: {logistic_precision:.4f}"
)

print(
    f"Recall   : {logistic_recall:.4f}"
)

print(
    f"F1 Score : {logistic_f1:.4f}"
)


# ============================================================
# 16. LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================================

cm_logistic = confusion_matrix(
    y_test,
    y_pred_logistic
)

print("\nLogistic Regression Confusion Matrix:")
print(cm_logistic)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.title(
    "Logistic Regression - Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_logistic.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 17. MODEL 2 - DECISION TREE
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 - DECISION TREE")
print("=" * 70)

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)

decision_tree.fit(
    X_train,
    y_train
)

y_pred_tree = decision_tree.predict(
    X_test
)

print("\nDecision Tree Predictions:")
print(y_pred_tree)


# ============================================================
# 18. DECISION TREE METRICS
# ============================================================

tree_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

tree_precision = precision_score(
    y_test,
    y_pred_tree,
    zero_division=0
)

tree_recall = recall_score(
    y_test,
    y_pred_tree,
    zero_division=0
)

tree_f1 = f1_score(
    y_test,
    y_pred_tree,
    zero_division=0
)

print("\nDecision Tree Evaluation:")

print(
    f"Accuracy : {tree_accuracy:.4f}"
)

print(
    f"Precision: {tree_precision:.4f}"
)

print(
    f"Recall   : {tree_recall:.4f}"
)

print(
    f"F1 Score : {tree_f1:.4f}"
)


# ============================================================
# 19. DECISION TREE CONFUSION MATRIX
# ============================================================

cm_tree = confusion_matrix(
    y_test,
    y_pred_tree
)

print("\nDecision Tree Confusion Matrix:")
print(cm_tree)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_tree,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.title(
    "Decision Tree - Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_decision_tree.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 20. CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORTS")
print("=" * 70)

print("\nLogistic Regression:")
print(
    classification_report(
        y_test,
        y_pred_logistic,
        labels=[0, 1],
        target_names=["No", "Yes"],
        zero_division=0
    )
)

print("\nDecision Tree:")
print(
    classification_report(
        y_test,
        y_pred_tree,
        labels=[0, 1],
        target_names=["No", "Yes"],
        zero_division=0
    )
)


# ============================================================
# 21. MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],

    "Accuracy": [
        logistic_accuracy,
        tree_accuracy
    ],

    "Precision": [
        logistic_precision,
        tree_precision
    ],

    "Recall": [
        logistic_recall,
        tree_recall
    ],

    "F1 Score": [
        logistic_f1,
        tree_f1
    ]
})


print("\n" + "=" * 70)
print("SUPERVISED MODEL COMPARISON")
print("=" * 70)

print(
    comparison.round(4).to_string(
        index=False
    )
)


# ============================================================
# 22. SAVE COMPARISON TABLE
# ============================================================

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 23. MODEL COMPARISON GRAPH
# ============================================================

comparison_graph = comparison.set_index(
    "Model"
)

comparison_graph.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Supervised Model Performance Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
)

plt.xticks(
    rotation=0
)

plt.ylim(
    0,
    1.1
)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 24. DECISION TREE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(18, 10)
)

plot_tree(
    decision_tree,
    feature_names=feature_columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True
)

plt.title(
    "Decision Tree Visualization"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 25. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("MODEL 3 - K-MEANS CLUSTERING")
print("=" * 70)

print(
    "\nPurchase target is NOT used for clustering."
)

clustering_features = df[
    feature_columns
]

cluster_scaler = StandardScaler()

X_cluster_scaled = cluster_scaler.fit_transform(
    clustering_features
)


# ============================================================
# 26. K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    X_cluster_scaled
)

df["Cluster"] = clusters


print("\nCluster Distribution:")
print(
    df["Cluster"].value_counts().sort_index()
)


# ============================================================
# 27. CLUSTER VS PURCHASE
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTER VS PURCHASE ANALYSIS")
print("=" * 70)

cluster_purchase_count = pd.crosstab(
    df["Cluster"],
    df["Purchase"]
)

# Ensure both target columns exist
cluster_purchase_count = cluster_purchase_count.reindex(
    columns=[0, 1],
    fill_value=0
)

cluster_purchase_count.columns = [
    "No",
    "Yes"
]

print("\nCluster vs Purchase Count:")
print(cluster_purchase_count)


# ============================================================
# 28. CLUSTER PURCHASE PERCENTAGE
# ============================================================

cluster_purchase_percentage = pd.crosstab(
    df["Cluster"],
    df["Purchase"],
    normalize="index"
) * 100

cluster_purchase_percentage = (
    cluster_purchase_percentage
    .reindex(
        columns=[0, 1],
        fill_value=0
    )
)

cluster_purchase_percentage.columns = [
    "No (%)",
    "Yes (%)"
]

print("\nCluster vs Purchase Percentage:")
print(
    cluster_purchase_percentage.round(2)
)


# ============================================================
# 29. SAVE CLUSTER ANALYSIS
# ============================================================

cluster_purchase_count.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_purchase_count.csv"
    )
)

cluster_purchase_percentage.round(
    2
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_purchase_percentage.csv"
    )
)


# ============================================================
# 30. K-MEANS VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 6)
)

scatter = plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel(
    "Annual Income"
)

plt.ylabel(
    "Spending Score"
)

plt.title(
    "K-Means Customer Clusters"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_clusters.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 31. CLUSTER DISTRIBUTION GRAPH
# ============================================================

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title(
    "Customer Distribution Across K-Means Clusters"
)

plt.xlabel(
    "Cluster"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cluster_distribution.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 32. FINAL MODEL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL RECOMMENDATION")
print("=" * 70)


# F1 Score is used as the primary metric.
# Recall is used as a tie-breaker.

if logistic_f1 > tree_f1:

    recommended_model = "Logistic Regression"

elif tree_f1 > logistic_f1:

    recommended_model = "Decision Tree"

else:

    if logistic_recall >= tree_recall:
        recommended_model = "Logistic Regression"
    else:
        recommended_model = "Decision Tree"


print(
    f"\nRecommended Model: {recommended_model}"
)

print(
    "\nReason:"
)

if recommended_model == "Logistic Regression":

    print(
        "Logistic Regression achieved the better "
        "F1 Score among the supervised models."
    )

else:

    print(
        "Decision Tree achieved the better "
        "F1 Score among the supervised models."
    )


# ============================================================
# 33. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print("""
ML-1 - Fundamentals:
- Dataset understanding
- Data cleaning
- Feature and target selection
- Feature scaling
- Train/test split

ML-2 - Supervised Learning:
- Logistic Regression
- Decision Tree

ML-3 - Unsupervised Learning:
- K-Means Clustering

ML-4 - Evaluation:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Model comparison

K-Means Analysis:
- Customer clusters were created without using Purchase.
- Clusters were compared with actual Purchase behavior.
""")


# ============================================================
# 34. OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("GENERATED OUTPUT FILES")
print("=" * 70)

print("""
output/
│
├── confusion_matrix_logistic.png
├── confusion_matrix_decision_tree.png
├── model_comparison.csv
├── model_comparison.png
├── decision_tree.png
├── kmeans_clusters.png
├── cluster_distribution.png
├── cluster_purchase_count.csv
└── cluster_purchase_percentage.csv
""")


# ============================================================
# 35. LIMITATION
# ============================================================

print("\n" + "=" * 70)
print("PROJECT LIMITATION")
print("=" * 70)

print("""
The dataset contains only 20 observations.
Therefore, the test set contains only a small number
of records and the evaluation metrics may vary
significantly with different train/test splits.

The results should therefore be considered an
educational model comparison rather than a
production-ready prediction system.
""")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY! ✅")
print("=" * 70)