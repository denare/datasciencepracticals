import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("==================================================")
    # 1. Load the Iris dataset (classic Data Science/ML dataset)
    # --------------------------------------------------
    print("1. Loading the Iris dataset...")
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # 2. Explore the dataset using Pandas
    # --------------------------------------------------
    print("\n2. Exploring data with Pandas:")
    print("--- First 5 rows of the dataset ---")
    print(df.head())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Target Class Distribution ---")
    # Mapping target integer labels to actual species names
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(species_map)
    print(df['species'].value_counts())
    
    # 3. Perform basic NumPy operations
    # --------------------------------------------------
    print("\n3. Performing NumPy operations:")
    # Extract feature values as a NumPy array
    features_array = df.iloc[:, :4].values
    
    mean_features = np.mean(features_array, axis=0)
    std_features = np.std(features_array, axis=0)
    
    print("Feature column names:", iris.feature_names)
    print("Mean of each feature:", mean_features)
    print("Std dev of each feature:", std_features)
    
    # 4. Data Visualization with Matplotlib & Seaborn
    # --------------------------------------------------
    print("\n4. Visualizing data...")
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Scatter plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='sepal length (cm)', 
        y='sepal width (cm)', 
        hue='species', 
        palette='viridis',
        s=70
    )
    plt.title('Iris Sepal Length vs Sepal Width')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.tight_layout()
    plt.savefig('plots/sepal_scatter.png')
    plt.close()
    print("Saved 'plots/sepal_scatter.png'")
    
    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    # Select only numeric columns for correlation
    numeric_df = df.drop(columns=['species', 'target'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    print("Saved 'plots/correlation_heatmap.png'")
    
    # 5. Machine Learning with Scikit-learn
    # --------------------------------------------------
    print("\n5. Training a Machine Learning Model (Random Forest Classifier)...")
    
    # Define features (X) and target (y)
    X = df.drop(columns=['target', 'species'])
    y = df['target']
    
    # Split the dataset: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Initialize and train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    print("==================================================")
    print("Ready to start learning! Try editing this script or launch a Jupyter notebook to experiment.")

if __name__ == "__main__":
    main()
