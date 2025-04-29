import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load and split the data
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Scale the features (Logistic Regression needs this)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split after scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train models
log_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier()
forest_model = RandomForestClassifier()

log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

# Predictions
log_preds = log_model.predict(X_test)
tree_preds = tree_model.predict(X_test)
forest_preds = forest_model.predict(X_test)

# Print results
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, log_preds))
print(classification_report(y_test, log_preds))

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, tree_preds))
print(classification_report(y_test, tree_preds))

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, forest_preds))
print(classification_report(y_test, forest_preds))

# ROC-AUC scores
log_proba = log_model.predict_proba(X_test)[:, 1]
tree_proba = tree_model.predict_proba(X_test)[:, 1]
forest_proba = forest_model.predict_proba(X_test)[:, 1]

print("\n--- ROC-AUC Scores ---")
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, log_proba))
print("Decision Tree ROC-AUC:", roc_auc_score(y_test, tree_proba))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, forest_proba))

# Confusion Matrix Plots
for model_name, preds in zip(
    ["Logistic Regression", "Decision Tree", "Random Forest"],
    [log_preds, tree_preds, forest_preds]
):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

# Random Forest Feature Importance
features = df.drop("target", axis=1).columns
importances = forest_model.feature_importances_
forest_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=forest_importance, y=forest_importance.index)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

example_patients = pd.DataFrame([
    [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],   # Patient 1
    [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2],   # Patient 2
    [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2],   # Patient 3
    [56, 1, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 1, 2],   # Patient 4
    [57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 0, 1, 2],   # Patient 5
    [45, 1, 1, 110, 264, 0, 1, 132, 0, 1.2, 1, 0, 2],   # Patient 6
    [54, 1, 0, 140, 239, 0, 1, 160, 0, 1.2, 0, 0, 2],   # Patient 7
    [39, 0, 2, 138, 220, 0, 1, 152, 0, 0.0, 2, 0, 2],   # Patient 8
    [50, 1, 2, 140, 233, 0, 1, 163, 0, 0.6, 1, 1, 3],   # Patient 9
    [64, 0, 1, 130, 303, 0, 1, 122, 0, 2.0, 2, 0, 2],   # Patient 10
    [65, 0, 1, 130, 303, 0, 1, 122, 0, 2.0, 2, 0, 2],   # Patient 11
], columns=X.columns)

example_patients_scaled = scaler.transform(example_patients)
predictions = forest_model.predict(example_patients_scaled)
probs = forest_model.predict_proba(example_patients_scaled)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probs), 1):
    print(f"Patient {i}: {'Heart Disease' if pred == 1 else 'No Heart Disease'} (Probability: {prob:.2f})")