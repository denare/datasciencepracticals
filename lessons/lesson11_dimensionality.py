# %% [markdown]
# # Lesson 11: Feature Selection & PCA (Dimensionality Reduction) 📐
#
# High-dimensional datasets (datasets with many features) present several challenges:
# - **Overfitting:** The model memorizes noise instead of general patterns.
# - **Computational cost:** Training takes much longer.
# - **The Curse of Dimensionality:** In high dimensions, data points become extremely sparse.
#
# To solve this, we use two main strategies:
# 1. **Feature Selection:** Keeping only the most relevant original features (e.g., SelectKBest, VarianceThreshold).
# 2. **Feature Extraction (Dimensionality Reduction):** Transforming original features into a new, lower-dimensional space (e.g., Principal Component Analysis - PCA).
#
# In this lesson, we will cover both methods using Scikit-Learn.

# %%
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## 1. Setting Up the Data
# We will use the Breast Cancer dataset, which contains 30 numerical features.

# %%
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Original shape: {X_train.shape}")

# %% [markdown]
# ## 2. Feature Selection
# Let's explore two common ways to filter out features:
#
# ### A. Variance Threshold (removing near-constant features)
# If a feature has the exact same value for almost every row, it has zero variance and provides no predictive power.

# %%
# Remove features that have a variance lower than 0.01 (meaning they are mostly constant)
# Note: Usually we scale the data first or analyze variance on the original scale with caution.
selector_var = VarianceThreshold(threshold=0.01)
X_train_var = selector_var.fit_transform(X_train)

print(f"Shape after VarianceThreshold: {X_train_var.shape}")
print(f"Features removed: {X_train.shape[1] - X_train_var.shape[1]}")

# %% [markdown]
# ### B. SelectKBest (univariate feature selection)
# SelectKBest selects the $K$ features that have the strongest individual statistical relationship with the target class.
# Here we use `f_classif` (ANOVA F-value) as our scoring function for classification.

# %%
# Select top 10 features
selector_kbest = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = selector_kbest.fit_transform(X_train, y_train)

# Get the names of the selected features
selected_indices = selector_kbest.get_support(indices=True)
selected_features = X_train.columns[selected_indices]

print(f"Shape after SelectKBest: {X_train_kbest.shape}")
print("\nTop 10 Selected Features:")
print(list(selected_features))

# %% [markdown]
# ## 3. Dimensionality Reduction with PCA
# Unlike feature selection which picks a subset of features, **PCA (Principal Component Analysis)** projects the features into a new space.
# The new features (called **Principal Components**) are linear combinations of the original features, ordered by how much variance they explain.
#
# **Crucial:** You MUST scale your data before applying PCA! PCA is highly sensitive to the scale of the features.

# %%
# 1. Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Fit PCA (keep all components first to see how much variance they explain)
pca_full = PCA()
pca_full.fit(X_train_scaled)

# 3. Plot the Cumulative Explained Variance (Scree Plot)
plt.figure(figsize=(8, 4))
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle=':', label='95% Explained Variance')
plt.title('PCA Explained Variance Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.savefig('plots/lesson11_pca_scree.png')
plt.close()

print("Saved scree plot to 'plots/lesson11_pca_scree.png'")

# Print the variance explained by the first few components
for i, ratio in enumerate(pca_full.explained_variance_ratio_[:5]):
    print(f"Principal Component {i+1} explains {ratio*100:.2f}% of the variance.")
print(f"Total variance explained by first 5 components: {cumulative_variance[4]*100:.2f}%")

# %% [markdown]
# **Notice:** The first 5 principal components capture over 84% of the information (variance) in all 30 original features!

# %% [markdown]
# ## 4. Training a Model on PCA Components
# Let's write a Pipeline that standardizes the data, applies PCA (keeping only 5 components), and trains a Random Forest.

# %%
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('classifier', RandomForestClassifier(random_state=42))
])

pca_pipeline.fit(X_train, y_train)
y_pred_pca = pca_pipeline.predict(X_test)
pca_acc = accuracy_score(y_test, y_pred_pca)

# Train a model on all 30 features (no PCA) for comparison
normal_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
normal_pipeline.fit(X_train, y_train)
normal_acc = accuracy_score(y_test, normal_pipeline.predict(X_test))

print("--- 4. PCA Pipeline vs Normal Pipeline ---")
print(f"Accuracy with all 30 features: {normal_acc * 100:.2f}%")
print(f"Accuracy with only 5 PCA components: {pca_acc * 100:.2f}%")

# %% [markdown]
# ## YOUR TURN! (Exercises)
# In this exercise, you will build a pipeline that combines `SelectKBest` with a classifier and compare it.

# %%
# TODO: Create a pipeline that standardizes features, selects top 5 features with SelectKBest, and runs RandomForest
# pipeline_kbest = Pipeline([
#     ('scaler', StandardScaler()),
#     ('select', SelectKBest(score_func=f_classif, k=5)),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])

# TODO: Fit the pipeline on X_train and y_train
# pipeline_kbest.fit(...)

# TODO: Predict on X_test and print the test accuracy
# kbest_acc = accuracy_score(y_test, pipeline_kbest.predict(X_test))
# print(f"Exercise Accuracy (SelectKBest=5): {kbest_acc * 100:.2f}%")
