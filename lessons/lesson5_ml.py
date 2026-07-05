# %% [markdown]
# # Lesson 5: Introduction to Machine Learning 🤖
# Machine Learning (ML) is the process of using algorithms to parse data, learn from it,
# and then make a determination or prediction about something in the world.
#
# In this lesson, we will use **Scikit-learn**, the most popular ML library in Python.
# We will build a Classification model using a breast cancer dataset:
# 1. Loading the Data
# 2. Preparing Features (X) and Target (y)
# 3. Train/Test Split
# 4. Training a Machine Learning Model (Logistic Regression)
# 5. Evaluating Model Performance (Accuracy, Confusion Matrix)
# 6. Exercises for you!

# %%
import os
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    print("=== Intro to Machine Learning Lesson ===")
    os.makedirs('plots', exist_ok=True)
    
# %% [markdown]
    # -------------------------------------------------------------------------
    # 1. Loading the Dataset
    # -------------------------------------------------------------------------
# %%
    print("\n--- 1. Loading the Dataset ---")
    # Load breast cancer dataset from scikit-learn
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    print(df.head(3))
    print("\nTarget classes:", data.target_names)
    print("Class distribution:\n", df['target'].value_counts())

# %% [markdown]
    # -------------------------------------------------------------------------
    # 2. Preparing Features (X) and Target (y)
    # -------------------------------------------------------------------------
# %%
    print("\n--- 2. Preparing Features and Target ---")
    # Features (X) are all columns except the target
    X = df.drop(columns=['target'])
    
    # Target (y) is what we want to predict
    y = df['target']
    
    print(f"X shape (Features): {X.shape}")
    print(f"y shape (Target): {y.shape}")

# %% [markdown]
    # -------------------------------------------------------------------------
    # 3. Train/Test Split
    # -------------------------------------------------------------------------
# %%
    print("\n--- 3. Train/Test Split ---")
    # We split the data so we can test the model on unseen data
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

# %% [markdown]
    # -------------------------------------------------------------------------
    # 4. Training the Model
    # -------------------------------------------------------------------------
# %%
    print("\n--- 4. Training the Model (Logistic Regression) ---")
    # Initialize the model (max_iter=10000 ensures it has enough time to converge)
    model = LogisticRegression(max_iter=10000, random_state=42)
    
    # Train (fit) the model on the training data
    model.fit(X_train, y_train)
    print("Model training complete!")

# %% [markdown]
    # -------------------------------------------------------------------------
    # 5. Evaluating Model Performance
    # -------------------------------------------------------------------------
# %%
    print("\n--- 5. Evaluating the Model ---")
    # Ask the trained model to predict the labels for the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    # Detailed Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Confusion Matrix (True Positives, False Positives, etc.)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotting the Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data.target_names,
                yticklabels=data.target_names)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('plots/lesson5_confusion_matrix.png')
    plt.close()
    print("Saved 'plots/lesson5_confusion_matrix.png'")

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
    
    # Exercise 1: Train a different model!
    # Let's try a Random Forest Classifier instead of Logistic Regression.
    from sklearn.ensemble import RandomForestClassifier
    
    # TODO: Initialize RandomForestClassifier with random_state=42
    rf_model = None
    
    # TODO: Fit the rf_model using X_train and y_train
    # rf_model.fit(...)
    
    # TODO: Predict using X_test
    rf_pred = None
    
    # TODO: Calculate accuracy of the Random Forest
    rf_accuracy = None
    
    print(f"Exercise 1: Random Forest Accuracy: {rf_accuracy}")

if __name__ == "__main__":
    main()
