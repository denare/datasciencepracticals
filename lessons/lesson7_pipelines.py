# %% [markdown]
# # Lesson 7: Feature Scaling & Pipelines 🏭
# 
# Welcome to Level 2! In the capstone project, you performed data cleaning 
# and model training manually, step-by-step. 
#
# In the real world, this can get messy and lead to **Data Leakage** (accidentally 
# using information from your test set during training). 
#
# This lesson introduces two critical concepts for production ML code:
# 1. **Feature Scaling**: Normalizing features so they have the same scale (e.g. 0 to 1).
# 2. **Pipelines**: Chaining preprocessing steps and your ML model into a single object.

# %%
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def main():
    print("=== Feature Scaling & Pipelines Lesson ===")
    os.makedirs('plots', exist_ok=True)
    
# %% [markdown]
    # -------------------------------------------------------------------------
    # 1. Why Feature Scaling Matters
    # -------------------------------------------------------------------------
# %%
    print("\n--- 1. Why Feature Scaling Matters ---")
    
    # We will use the Wine dataset. 
    # It classifies 3 types of wine based on 13 chemical features.
    wine = load_wine(as_frame=True)
    df = wine.frame
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Let's look at the first two features: Alcohol and Proline
    print(df[['alcohol', 'proline']].head())
    
    print("\nNotice the scales:")
    print(f"Alcohol ranges from ~{df['alcohol'].min():.1f} to {df['alcohol'].max():.1f}")
    print(f"Proline ranges from ~{df['proline'].min():.1f} to {df['proline'].max():.1f}")
    
    # ML models like KNN use Distance. If we don't scale, 'proline' (which is in the 1000s) 
    # will completely dominate 'alcohol' (which is around 13).
    
    # Let's visualize the unscaled data
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='alcohol', y='proline', hue='target', palette='Set1')
    plt.title('Unscaled Features')
    
    # Now let's apply StandardScaler (makes mean=0, std=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['alcohol', 'proline']])
    df_scaled = pd.DataFrame(scaled_features, columns=['alcohol_scaled', 'proline_scaled'])
    df_scaled['target'] = df['target']
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_scaled, x='alcohol_scaled', y='proline_scaled', hue='target', palette='Set1')
    plt.title('Standard Scaled Features')
    
    plt.tight_layout()
    plt.savefig('plots/lesson7_scaling_comparison.png')
    plt.close()
    print("\nSaved 'plots/lesson7_scaling_comparison.png'")
    
# %% [markdown]
    # -------------------------------------------------------------------------
    # 2. Training WITHOUT Scaling
    # -------------------------------------------------------------------------
# %%
    print("\n--- 2. Training WITHOUT Scaling ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize K-Nearest Neighbors (KNN is highly sensitive to unscaled data)
    knn_unscaled = KNeighborsClassifier(n_neighbors=5)
    
    # Train and test
    knn_unscaled.fit(X_train, y_train)
    unscaled_pred = knn_unscaled.predict(X_test)
    unscaled_acc = accuracy_score(y_test, unscaled_pred)
    
    print(f"KNN Accuracy (Unscaled): {unscaled_acc * 100:.2f}%")
    
# %% [markdown]
    # -------------------------------------------------------------------------
    # 3. Enter: The Pipeline
    # -------------------------------------------------------------------------
# %%
    print("\n--- 3. Using a Pipeline for Scaling + Training ---")
    
    # Instead of scaling manually, we build a Pipeline.
    # A Pipeline chains together multiple steps. When we call .fit() on the pipeline, 
    # it applies StandardScaler.fit_transform() to the training data, and then passes 
    # the scaled data to KNeighborsClassifier.fit().
    # When we call .predict(), it applies StandardScaler.transform() and then predicts!
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),                 # Step 1: Scale the data
        ('classifier', KNeighborsClassifier(n_neighbors=5)) # Step 2: Train the model
    ])
    
    # Train the whole pipeline
    pipeline.fit(X_train, y_train)
    
    # Predict using the pipeline (it automatically scales X_test for us!)
    scaled_pred = pipeline.predict(X_test)
    scaled_acc = accuracy_score(y_test, scaled_pred)
    
    print(f"KNN Accuracy (With Pipeline & Scaling): {scaled_acc * 100:.2f}%")
    print(f"Improvement: +{(scaled_acc - unscaled_acc) * 100:.2f}%")

    # -------------------------------------------------------------------------
# %% [markdown]
    # YOUR TURN! (Exercises)
    # -------------------------------------------------------------------------
# %%
    print("\n==============================================")
    print("Now run the exercises below by calling them in main()!")
    print("==============================================")
    
    run_exercises(X_train, X_test, y_train, y_test)

def run_exercises(X_train, X_test, y_train, y_test):
    print("\n--- Running Exercises ---")
    
    # Exercise 1: Try a different scaler!
# %% [markdown]
    # Let's see how MinMaxScaler performs compared to StandardScaler.
    # MinMaxScaler scales all features to be exactly between 0 and 1.
    
    # TODO: Create a Pipeline using MinMaxScaler and KNeighborsClassifier
# %%
    minmax_pipeline = None
    
    # TODO: Fit the minmax_pipeline on the training data
    # minmax_pipeline.fit(...)
    
    # TODO: Predict on the test data and calculate accuracy
    minmax_acc = None
    
    print(f"Exercise 1: MinMaxScaler Pipeline Accuracy: {minmax_acc}")

if __name__ == "__main__":
    main()
