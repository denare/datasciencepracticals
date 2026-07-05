# %% [markdown]
# # Lesson 8: Hyperparameter Tuning 🎛️
#
# When training a Machine Learning model, we have two types of settings:
# 1. **Parameters**: Learned by the model during `.fit()` (e.g., weights in regression).
# 2. **Hyperparameters**: Settings we must choose *before* training (e.g., number of trees in Random Forest, or `k` in KNN).
#
# Finding the best combination of hyperparameters is called **Hyperparameter Tuning**.
# In this lesson, we will cover:
# - Grid Search (`GridSearchCV`): Tries every combination in a grid.
# - Randomized Search (`RandomizedSearchCV`): Tries random combinations (faster).
# - Tuning a Model inside a Pipeline.

import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("=== Hyperparameter Tuning Lesson ===")
    os.makedirs('plots', exist_ok=True)

    # Load Breast Cancer dataset
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------------------------------------------------
    # 1. Default Model Performance
    # -------------------------------------------------------------------------
    print("\n--- 1. Default Model Performance ---")
    # Initialize Random Forest with defaults
    rf_default = RandomForestClassifier(random_state=42)
    rf_default.fit(X_train, y_train)
    default_pred = rf_default.predict(X_test)
    default_acc = accuracy_score(y_test, default_pred)
    print(f"Default Random Forest Accuracy: {default_acc * 100:.2f}%")

    # -------------------------------------------------------------------------
    # 2. Grid Search (GridSearchCV)
    # -------------------------------------------------------------------------
    print("\n--- 2. Grid Search (GridSearchCV) ---")
    # Grid Search is exhaustive: it tries every single combination.
    # Here we define the parameter grid (2 * 3 * 2 = 12 total combinations):
    param_grid = {
        'n_estimators': [50, 100],          # Number of trees
        'max_depth': [3, 5, None],          # Maximum depth of tree
        'min_samples_split': [2, 5]         # Min samples required to split a node
    }

    rf = RandomForestClassifier(random_state=42)

    # cv=5 means 5-Fold Cross-Validation (evaluating each combination on 5 different folds)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    print("Searching the parameter grid...")
    grid_search.fit(X_train, y_train)

    print("\nBest Parameters found by Grid Search:")
    print(grid_search.best_params_)
    print(f"Best CV Score: {grid_search.best_score_ * 100:.2f}%")

    # Test the best estimator
    best_grid_model = grid_search.best_estimator_
    grid_pred = best_grid_model.predict(X_test)
    grid_acc = accuracy_score(y_test, grid_pred)
    print(f"Grid Search Best Model Test Accuracy: {grid_acc * 100:.2f}%")

    # -------------------------------------------------------------------------
    # 3. Randomized Search (RandomizedSearchCV)
    # -------------------------------------------------------------------------
    print("\n--- 3. Randomized Search (RandomizedSearchCV) ---")
    # If the grid is huge (e.g., thousands of combinations), Grid Search takes too long.
    # Randomized Search randomly samples combinations. It is much faster and often finds
    # just as good (or better) hyperparameters.
    
    from scipy.stats import randint

    # We can use distributions for continuous/integer parameters
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [3, 5, 10, None],
        'min_samples_split': randint(2, 10),
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)

    # n_iter=10 means try 10 random combinations
    random_search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_dist, n_iter=10, 
        cv=5, scoring='accuracy', random_state=42, n_jobs=-1
    )

    print("Searching a random sample of configurations...")
    random_search.fit(X_train, y_train)

    print("\nBest Parameters found by Randomized Search:")
    print(random_search.best_params_)
    print(f"Best CV Score: {random_search.best_score_ * 100:.2f}%")

    best_random_model = random_search.best_estimator_
    random_pred = best_random_model.predict(X_test)
    random_acc = accuracy_score(y_test, random_pred)
    print(f"Randomized Search Best Model Test Accuracy: {random_acc * 100:.2f}%")

    # -------------------------------------------------------------------------
    # 4. Tuning inside a Pipeline
    # -------------------------------------------------------------------------
    print("\n--- 4. Tuning Hyperparameters in a Pipeline ---")
    # What if we have scaling (or other preprocessing) steps in a Pipeline?
    # We can tune parameters of both the scaler AND the model!
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # To specify parameters for steps in a pipeline, use the syntax:
    # stepname__parametername (note the double underscore __)
    pipe_param_grid = {
        'scaler__with_mean': [True, False],
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [3, 5]
    }

    grid_pipe = GridSearchCV(pipe, pipe_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_pipe.fit(X_train, y_train)

    print("Best Pipeline Parameters:")
    print(grid_pipe.best_params_)
    
    pipe_acc = accuracy_score(y_test, grid_pipe.predict(X_test))
    print(f"Tuned Pipeline Test Accuracy: {pipe_acc * 100:.2f}%")

    # -------------------------------------------------------------------------
    # YOUR TURN! (Exercises)
    # -------------------------------------------------------------------------
    print("\n==============================================")
    print("Now run the exercises below by calling them in main()!")
    print("==============================================")
    
    run_exercises(X_train, X_test, y_train, y_test)

def run_exercises(X_train, X_test, y_train, y_test):
    print("\n--- Running Exercises ---")
    
    # Exercise 1: Tune a Support Vector Classifier (SVC)!
    from sklearn.svm import SVC

    # TODO: Create a pipeline with StandardScaler and SVC
    # svc_pipe = Pipeline(...)

    # TODO: Define a parameter grid to tune SVC parameters:
    # - 'svc__C': [0.1, 1, 10]
    # - 'svc__kernel': ['linear', 'rbf']
    # svc_param_grid = {...}

    # TODO: Perform GridSearchCV on the svc_pipe
    # svc_grid = GridSearchCV(...)
    # svc_grid.fit(X_train, y_train)

    # TODO: Print the best parameters and test accuracy
    best_params = None
    test_accuracy = None
    
    print(f"Exercise 1: Best SVC Params: {best_params}")
    print(f"Exercise 1: SVC Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
