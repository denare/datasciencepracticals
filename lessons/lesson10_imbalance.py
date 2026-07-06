# %% [markdown]
# # Lesson 10: Handling Imbalanced Data ⚖️
#
# In many real-world applications (fraud detection, rare disease diagnosis, spam filtering), 
# the dataset is highly imbalanced. For example, 99% of transactions are legitimate, and only 1% are fraud.
#
# If a model simply predicts "Legitimate" for every transaction, it achieves **99% accuracy**, but is 
# completely useless! 
#
# In this lesson, we will cover:
# 1. Why Accuracy is a misleading metric for imbalanced datasets.
# 2. Key evaluation metrics: Precision, Recall, F1-Score, and Confusion Matrix.
# 3. Fixing imbalance via model parameters (`class_weight='balanced'`).
# 4. Fixing imbalance via resampling: SMOTE (Synthetic Minority Over-sampling Technique).
# 5. Exercises.

# %%
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
# pyrefly: ignore [missing-import]
from imblearn.pipeline import Pipeline as ImbPipeline

# %% [markdown]
# ## 1. Setting Up an Imbalanced Dataset
# We will synthetically generate a dataset with a 95:5 ratio between Class 0 (majority) and Class 1 (minority).

# %%
# Generate 1000 samples, 20 features, 95% Class 0 and 5% Class 1
X, y = make_classification(
    n_samples=1000, n_features=10, n_classes=2, 
    weights=[0.95, 0.05], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("--- Class Distribution in Training Set ---")
print(pd.Series(y_train).value_counts(normalize=True))

# %% [markdown]
# ## 2. The Illusion of Accuracy
# Let's train a standard Random Forest Classifier and look at the accuracy vs. other metrics.

# %%
# Train a default Random Forest
model_default = RandomForestClassifier(random_state=42)
model_default.fit(X_train, y_train)
y_pred = model_default.predict(X_test)

print("--- 2. Default Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Let's check the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# Format:
# [[ TrueNegatives   FalsePositives ]
#  [ FalseNegatives  TruePositives  ]]

# Let's check the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# **Notice:** Even though the accuracy is ~95%, the Recall for the minority class (1) is very low! 
# The model is missing a lot of minority instances because it favors the majority class.

# %% [markdown]
# ## 3. Method 1: Class Weights
# Most Scikit-learn algorithms have a `class_weight` parameter. Setting it to `'balanced'` automatically 
# adjusts weights inversely proportional to class frequencies, penalizing mistakes on the minority class more.

# %%
# Train with balanced class weights
model_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)

print("--- 3. Performance with Class Weights ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_weighted) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_weighted))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_weighted))

# %% [markdown]
# ## 4. Method 2: Resampling with SMOTE
# **SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic (artificial) training samples 
# of the minority class by interpolating between existing minority samples.
#
# **Crucial rule:** Never apply SMOTE to the test set! It must only be done on the training folds.
# Using `imblearn.pipeline.Pipeline` guarantees SMOTE is only applied during training.

# %%
# Build an imbalanced pipeline that applies SMOTE during training, then fits a Classifier
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)
y_pred_smote = pipeline.predict(X_test)

print("--- 4. Performance with SMOTE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_smote) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_smote))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote))

# %% [markdown]
# ## YOUR TURN! (Exercises)
# In this exercise, you will train a Support Vector Classifier (SVC) using SMOTE.

# %%
from sklearn.svm import SVC

# TODO: Create an imbalanced-learn Pipeline with SMOTE and SVC
# pipeline_svc = ImbPipeline(...)

# TODO: Fit the pipeline on X_train and y_train
# pipeline_svc.fit(...)

# TODO: Predict on X_test and print the classification report
# svc_pred = pipeline_svc.predict(...)
# print(classification_report(y_test, svc_pred))
