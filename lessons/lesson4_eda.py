# %% [markdown]
# # Lesson 4: Exploratory Data Analysis (EDA) 🕵️‍♀️
# EDA is the process of performing initial investigations on data to discover patterns,
# spot anomalies, test hypotheses, and check assumptions with the help of summary
# statistics and graphical representations.
#
# In this lesson, we will explore the famous "Titanic" dataset:
# 1. Loading the dataset
# 2. Handling Missing Data
# 3. Univariate Analysis (Examining single variables)
# 4. Bivariate Analysis (Examining relationships between two variables)
# 5. Feature Engineering (Creating new features)
# 6. Exercises for you!

# %%
import os
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== Exploratory Data Analysis Lesson ===")
    os.makedirs('plots', exist_ok=True)
    
# %% [markdown]
    # -------------------------------------------------------------------------
    # 1. Loading the Dataset
    # -------------------------------------------------------------------------
# %%
    print("\n--- 1. Loading the Titanic Dataset ---")
    # Seaborn comes with several built-in datasets, Titanic is a classic
    df = sns.load_dataset('titanic')
    print(df.head())
    
    print("\nDataset Info:")
    df.info()

# %% [markdown]
    # -------------------------------------------------------------------------
    # 2. Handling Missing Data
    # -------------------------------------------------------------------------
# %%
    print("\n--- 2. Handling Missing Data ---")
    print("Missing values per column:\n", df.isnull().sum())
    
    # The 'deck' column has too many missing values, let's drop it
    df = df.drop(columns=['deck'])
    
    # Fill missing 'age' values with the median age
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age)
    
    # Fill missing 'embarked' and 'embark_town' with the most frequent value (mode)
    mode_embarked = df['embarked'].mode()[0]
    df['embarked'] = df['embarked'].fillna(mode_embarked)
    df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
    
    print("\nMissing values after cleaning:\n", df.isnull().sum())

# %% [markdown]
    # -------------------------------------------------------------------------
    # 3. Univariate Analysis
    # -------------------------------------------------------------------------
# %%
    print("\n--- 3. Univariate Analysis (Single Variable) ---")
    
    # Let's see the survival rate (0 = No, 1 = Yes)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='survived', hue='survived', palette='Set1', legend=False)
    plt.title('Survival Count (0 = No, 1 = Yes)')
    plt.savefig('plots/lesson4_survival_count.png')
    plt.close()
    print("Saved 'plots/lesson4_survival_count.png'")
    
    # Let's look at the age distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
    plt.title('Age Distribution of Passengers')
    plt.savefig('plots/lesson4_age_distribution.png')
    plt.close()
    print("Saved 'plots/lesson4_age_distribution.png'")

# %% [markdown]
    # -------------------------------------------------------------------------
    # 4. Bivariate Analysis
    # -------------------------------------------------------------------------
# %%
    print("\n--- 4. Bivariate Analysis (Two Variables) ---")
    
    # Survival rate by gender
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='sex', y='survived', hue='sex', palette='Pastel1', legend=False)
    plt.title('Survival Rate by Gender')
    plt.ylabel('Survival Probability')
    plt.savefig('plots/lesson4_survival_by_gender.png')
    plt.close()
    print("Saved 'plots/lesson4_survival_by_gender.png'")
    
    # Survival by Passenger Class
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='pclass', y='survived', hue='sex', palette='muted')
    plt.title('Survival Rate by Passenger Class and Gender')
    plt.ylabel('Survival Probability')
    plt.savefig('plots/lesson4_survival_by_class.png')
    plt.close()
    print("Saved 'plots/lesson4_survival_by_class.png'")
    
    # Age vs Fare scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.6, palette='coolwarm')
    plt.title('Age vs Fare (Colored by Survival)')
    plt.savefig('plots/lesson4_age_vs_fare.png')
    plt.close()
    print("Saved 'plots/lesson4_age_vs_fare.png'")

# %% [markdown]
    # -------------------------------------------------------------------------
    # 5. Feature Engineering
    # -------------------------------------------------------------------------
# %%
    print("\n--- 5. Feature Engineering ---")
    # Create a new feature: 'family_size' = siblings/spouses (sibsp) + parents/children (parch) + 1 (self)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    print("Created 'family_size' feature. Head:\n", df[['sibsp', 'parch', 'family_size']].head())
    
    # Create a categorical 'is_alone' feature based on family size
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    print("\nCreated 'is_alone' feature. Head:\n", df[['family_size', 'is_alone']].head())
    
    # -------------------------------------------------------------------------
# %% [markdown]
    # YOUR TURN! (Exercises)
    # -------------------------------------------------------------------------
# %%
    print("\n==============================================")
    print("Now run the exercises below by calling them in main()!")
    print("==============================================")
    
    run_exercises(df)

def run_exercises(df):
    print("\n--- Running Exercises ---")
    
    # Exercise 1: Calculate the overall survival rate (mean of 'survived').
    # Hint: Use df['survived'].mean()
    # TODO: Write your code here
    survival_rate = None
    print(f"Exercise 1 Survival Rate: {survival_rate}")
    
    # Exercise 2: Create a plot to see if 'is_alone' affects the survival rate.
    # Hint: Use sns.barplot(data=df, x='is_alone', y='survived')
    # Save the plot as 'plots/exercise_survival_alone.png'
    # TODO: Write your code here
    
    print("Exercise 2 plot generated (Check plots/exercise_survival_alone.png)")

if __name__ == "__main__":
    main()
