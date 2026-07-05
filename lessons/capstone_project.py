# %% [markdown]
# # Capstone Project: Titanic Survival Prediction 🚢
# 
# Congratulations on completing all 5 lessons! In this capstone project you will
# apply EVERY skill you have learned in one end-to-end data science workflow:
#
# ┌─────────────────────────────────────────────────────┐
# │  STEP 1: Load & Explore the Data (EDA)              │
# │  STEP 2: Clean & Handle Missing Values              │
# │  STEP 3: Feature Engineering                        │
# │  STEP 4: Encode Categorical Features                │
# │  STEP 5: Train/Test Split                           │
# │  STEP 6: Train & Compare 3 ML Models                │
# │  STEP 7: Evaluate the Best Model                    │
# └─────────────────────────────────────────────────────┘
#
# Dataset: Titanic (Will passengers survive? 0=No, 1=Yes)

import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style='whitegrid')

    # =========================================================================
    # STEP 1: Load & Explore the Data
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Load & Explore the Data")
    print("=" * 60)
    
    df = sns.load_dataset('titanic')
    print(f"Shape: {df.shape}")
    print(f"\nColumn list: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Quick EDA visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.countplot(data=df, x='survived', hue='sex', palette='Set2', ax=axes[0])
    axes[0].set_title('Survival Count by Gender')
    axes[0].set_xlabel('Survived (0=No, 1=Yes)')
    
    sns.barplot(data=df, x='pclass', y='survived', hue='pclass',
                palette='muted', legend=False, ax=axes[1])
    axes[1].set_title('Survival Rate by Passenger Class')
    axes[1].set_xlabel('Passenger Class')
    axes[1].set_ylabel('Survival Probability')
    
    plt.tight_layout()
    plt.savefig('plots/capstone_eda.png')
    plt.close()
    print("\nSaved EDA chart → 'plots/capstone_eda.png'")

    # =========================================================================
    # STEP 2: Clean & Handle Missing Values
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Clean & Handle Missing Values")
    print("=" * 60)
    
    # Drop columns with too many missing values or redundant info
    df = df.drop(columns=['deck', 'embark_town', 'who', 'alive', 'adult_male', 'class'])
    
    # Fill missing 'age' with median
    df['age'] = df['age'].fillna(df['age'].median())
    
    # Fill missing 'embarked' with the most common port
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    print(f"Missing values after cleaning:\n{df.isnull().sum()}")
    print(f"Cleaned shape: {df.shape}")

    # =========================================================================
    # STEP 3: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)
    
    # Create new informative features from existing ones
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                              labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    print("New features added: 'family_size', 'is_alone', 'age_group'")
    print(df[['family_size', 'is_alone', 'age_group']].head())

    # =========================================================================
    # STEP 4: Encode Categorical Features
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Encode Categorical Features")
    print("=" * 60)
    
    # ML models need numbers, not strings
    le = LabelEncoder()
    
    df['sex'] = le.fit_transform(df['sex'])           # male=1, female=0
    df['embarked'] = le.fit_transform(df['embarked']) # C=0, Q=1, S=2
    df['age_group'] = le.fit_transform(df['age_group'].astype(str))
    
    print("Categorical columns encoded to integers.")

    # =========================================================================
    # STEP 5: Train/Test Split
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Train/Test Split")
    print("=" * 60)
    
    # Final feature set
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
                'embarked', 'family_size', 'is_alone', 'age_group']
    
    X = df[features]
    y = df['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training: {X_train.shape[0]} samples | Testing: {X_test.shape[0]} samples")

    # =========================================================================
    # STEP 6: Train & Compare 3 ML Models
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Train & Compare 3 ML Models")
    print("=" * 60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    results = {}
    for name, model in models.items():
        # Cross-validation gives a more reliable accuracy estimate
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {'cv_mean': cv_scores.mean(), 'test_acc': test_acc, 'model': model}
        print(f"  {name:<25} | CV Accuracy: {cv_scores.mean()*100:.2f}% | Test Accuracy: {test_acc*100:.2f}%")

    # =========================================================================
    # STEP 7: Evaluate the Best Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Evaluate the Best Model")
    print("=" * 60)
    
    # Pick the model with the highest test accuracy
    best_name = max(results, key=lambda n: results[n]['test_acc'])
    best_model = results[best_name]['model']
    y_pred = best_model.predict(X_test)
    
    print(f"🏆 Best Model: {best_name} ({results[best_name]['test_acc']*100:.2f}% test accuracy)\n")
    print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))
    
    # Side-by-side: Model comparison bar chart + Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart comparing models
    model_names = list(results.keys())
    test_accs = [results[n]['test_acc'] * 100 for n in model_names]
    colors = ['#2196F3' if n != best_name else '#4CAF50' for n in model_names]
    bars = axes[0].bar(model_names, test_accs, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_ylim([70, 100])
    axes[0].set_title('Model Accuracy Comparison', fontsize=13)
    axes[0].set_ylabel('Test Accuracy (%)')
    for bar, acc in zip(bars, test_accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Confusion matrix for best model
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Did Not Survive', 'Survived'],
                yticklabels=['Did Not Survive', 'Survived'],
                ax=axes[1])
    axes[1].set_title(f'Confusion Matrix — {best_name}', fontsize=13)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('plots/capstone_results.png')
    plt.close()
    print("\nSaved results chart → 'plots/capstone_results.png'")
    
    # Feature Importance (only for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(9, 5))
        sns.barplot(data=importance_df, x='Importance', y='Feature',
                    hue='Feature', palette='viridis', legend=False)
        plt.title(f'Feature Importance — {best_name}')
        plt.tight_layout()
        plt.savefig('plots/capstone_feature_importance.png')
        plt.close()
        print("Saved feature importance → 'plots/capstone_feature_importance.png'")
        print(f"\nTop 3 Most Important Features:\n{importance_df.head(3).to_string(index=False)}")
    
    print("\n" + "=" * 60)
    print("🎉 CAPSTONE PROJECT COMPLETE! Congratulations!")
    print("You have applied: EDA, Data Cleaning, Feature Engineering,")
    print("Encoding, Train/Test Split, Model Comparison & Evaluation.")
    print("=" * 60)

if __name__ == "__main__":
    main()
