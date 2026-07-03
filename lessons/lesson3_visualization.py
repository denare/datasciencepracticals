# %% [markdown]
# # Lesson 3: Data Visualization with Matplotlib & Seaborn 📊
# Data visualization is a critical part of data science. It allows us to:
# 1. Identify patterns, trends, and outliers in data.
# 2. Communicate findings clearly to stakeholders.
#
# In this lesson, we will explore:
# 1. **Matplotlib**: The standard plotting library. It is low-level and highly customizable.
# 2. **Seaborn**: A high-level library built on top of Matplotlib. It makes beautiful statistical graphics easy.
#
# Let's explore:
# 1. Figure & Axes structure in Matplotlib
# 2. Line & Scatter plots
# 3. Bar charts & Histograms
# 4. Box plots & Heatmaps in Seaborn
# 5. Exercises for you!

import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== Data Visualization Lesson ===")
    
    # -------------------------------------------------------------------------
    # 1. Setup & Sample Data
    # -------------------------------------------------------------------------
    os.makedirs('plots', exist_ok=True)
    
    # Synthetic time-series data
    days = np.arange(1, 11)
    sales_a = np.array([100, 120, 130, 115, 140, 160, 155, 170, 190, 210])
    sales_b = np.array([80, 95, 110, 120, 115, 130, 145, 150, 160, 185])
    
    # Load our sales CSV dataset for Seaborn examples
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df_sales = pd.read_csv(os.path.join(current_dir, 'sample_sales.csv'))
    df_sales['Revenue'] = df_sales['Quantity'] * df_sales['Price']

    # -------------------------------------------------------------------------
    # 2. Matplotlib: Line Plots & Customization
    # -------------------------------------------------------------------------
    print("\n1. Generating line plot using Matplotlib...")
    # Best practice: Use subplots to define fig (canvas) and ax (plot area)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(days, sales_a, label='Store A', color='blue', marker='o', linestyle='-')
    ax.plot(days, sales_b, label='Store B', color='orange', marker='s', linestyle='--')
    
    # Customizing labels and titles
    ax.set_title('Daily Sales comparison (Store A vs Store B)', fontsize=14, pad=15)
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Sales ($)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/lesson3_line_plot.png')
    plt.close()
    print("Saved 'plots/lesson3_line_plot.png'")

    # -------------------------------------------------------------------------
    # 3. Matplotlib: Scatter Plots
    # -------------------------------------------------------------------------
    print("\n2. Generating scatter plot using Matplotlib...")
    # Generate random points
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(x, y, c=y, cmap='viridis', alpha=0.8, edgecolors='none')
    
    # Add colorbar
    fig.colorbar(scatter, ax=ax, label='Value Intensity')
    
    ax.set_title('Scatter Plot with Viridis Colormap')
    ax.set_xlabel('X Variable')
    ax.set_ylabel('Y Variable')
    
    plt.savefig('plots/lesson3_scatter_plot.png')
    plt.close()
    print("Saved 'plots/lesson3_scatter_plot.png'")

    # -------------------------------------------------------------------------
    # 4. Seaborn: Histograms & Density Plots (KDE)
    # -------------------------------------------------------------------------
    print("\n3. Generating distribution plots using Seaborn...")
    # Seaborn automatically integrates with Pandas DataFrames
    sns.set_theme(style="whitegrid") # Sets Seaborn style globally
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_sales, x='Price', kde=True, bins=5, color='purple')
    plt.title('Distribution of Product Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    
    plt.savefig('plots/lesson3_price_distribution.png')
    plt.close()
    print("Saved 'plots/lesson3_price_distribution.png'")

    # -------------------------------------------------------------------------
    # 5. Seaborn: Categorical Plots (Bar & Box plots)
    # -------------------------------------------------------------------------
    print("\n4. Generating categorical plots using Seaborn...")
    # Bar plot showing mean price per category
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_sales, x='Category', y='Revenue', hue='Category', palette='Set2', legend=False)
    plt.title('Average Revenue per Product Category')
    plt.xlabel('Category')
    plt.ylabel('Average Revenue ($)')
    
    plt.savefig('plots/lesson3_category_revenue.png')
    plt.close()
    print("Saved 'plots/lesson3_category_revenue.png'")
    
    # Box plot showing range and quartiles of price per category
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_sales, x='Category', y='Price', hue='Category', palette='pastel', legend=False)
    plt.title('Price Distribution by Category')
    
    plt.savefig('plots/lesson3_price_boxplot.png')
    plt.close()
    print("Saved 'plots/lesson3_price_boxplot.png'")

    # -------------------------------------------------------------------------
    # YOUR TURN! (Exercises)
    # Run the script first, then write your code in the functions below.
    # -------------------------------------------------------------------------
    print("\n==============================================")
    print("Now run the exercises below by calling them in main()!")
    print("==============================================")
    
    run_exercises(df_sales)

def run_exercises(df):
    print("\n--- Running Exercises ---")
    
    # Exercise 1: Draw a simple line plot showing temperature over a week.
    # Days: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    # Temp: [22, 24, 21, 25, 27, 26, 23]
    # Save your plot as 'plots/exercise_temp.png'
    # TODO: Write your code here
    
    print("Exercise 1 line plot generated (Check plots/exercise_temp.png)")
    
    # Exercise 2: Draw a Seaborn bar plot showing the total revenue (y) generated by each Country (x).
    # Hint: Use sns.barplot(data=df, x='Country', y='Revenue', estimator=sum, errorbar=None)
    # Save your plot as 'plots/exercise_country_revenue.png'
    # TODO: Write your code here
    
    print("Exercise 2 bar plot generated (Check plots/exercise_country_revenue.png)")

if __name__ == "__main__":
    main()
