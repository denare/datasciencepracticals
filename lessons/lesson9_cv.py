# %% [markdown]
# # Lesson 9: Cross-Validation (CV) 🔄
#
# When we train a model, a simple train/test split can be risky. Depending on the random state, 
# our test set might end up being particularly easy or particularly hard, giving us a biased estimate of performance.
#
# **Cross-Validation** solves this by splitting the training data into $K$ equal parts (folds).
# The model is trained on $K-1$ folds and tested on the remaining fold. This is repeated $K$ times,
# and the average score is computed.
#
# In this lesson, we will cover:
# 1. K-Fold Cross-Validation.
# 2. Stratified K-Fold (for imbalanced classification).
# 3. Cross-Validation inside a Pipeline (preventing data leakage!).
# 4. Exercises.

# %%
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# %% [markdown]
# ## 1. K-Fold Cross-Validation
# Let's start with a standard K-Fold CV on the Iris dataset.

# %%
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Initialize model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# cv=5 means 5-Fold Cross-Validation
# cross_val_score automatically handles splitting, fitting, and scoring
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("--- 1. Standard 5-Fold CV ---")
print(f"Scores for each fold: {scores}")
print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std() * 100:.2f}%")

# %% [markdown]
# ## 2. Stratified K-Fold CV
# Standard K-Fold splits randomly. If the dataset has class imbalance, one fold might end up 
# with zero positive samples!
# **Stratified K-Fold** ensures that each fold contains roughly the same percentage of samples 
# of each target class.

# %%
# Let's define a specific K-Fold object to control shuffle and random_state
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\n--- 2. K-Fold vs Stratified K-Fold ---")
print(f"Shuffled K-Fold Mean Accuracy: {scores_kf.mean() * 100:.2f}%")
print(f"Stratified K-Fold Mean Accuracy: {scores_skf.mean() * 100:.2f}%")

# %% [markdown]
# ## 3. Cross-Validation with a Pipeline (No Data Leakage)
# **Crucial Concept:** If you scale your data BEFORE cross-validation, your scaler fits on the entire dataset.
# The mean/std of the validation fold "leaks" into the training fold!
#
# To prevent this, we must put the scaler and classifier inside a **Pipeline**, and pass the pipeline to `cross_val_score`.
# This ensures that for every fold, the scaler is only fitted on the training folds and applied to the validation fold.

# %%
# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

# Pass the pipeline to cross_val_score
scores_pipe = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')

print("\n--- 3. CV on a Preprocessing Pipeline ---")
print(f"Pipeline CV Mean Accuracy: {scores_pipe.mean() * 100:.2f}%")

# %% [markdown]
# ## YOUR TURN! (Exercises)
# In this exercise, you will run a 5-Fold Cross-Validation on a Support Vector Classifier (SVC).

# %%
from sklearn.svm import SVC

# TODO: Create a pipeline with StandardScaler and SVC
# pipeline_svc = Pipeline(...)

# TODO: Run a 5-Fold Stratified CV on this pipeline using the skf object defined above
# cv_scores = cross_val_score(...)

# TODO: Print the mean and standard deviation of the scores
# print(f"Exercise Mean Accuracy: {cv_scores.mean() * 100:.2f}%")
