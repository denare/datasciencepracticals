# =============================================================================
# DEEP DIVE 4: Seaborn Complete Reference 📊
# =============================================================================
# Seaborn is a high-level visualization library built on Matplotlib.
# It makes beautiful statistical plots with minimal code.
#
# This script covers EVERYTHING you need:
#   Part A: Themes & Styles
#   Part B: Distribution Plots (histplot, kdeplot, rugplot)
#   Part C: Categorical Plots (countplot, barplot, boxplot, violinplot, swarmplot)
#   Part D: Relational Plots (scatterplot, lineplot)
#   Part E: Heatmaps
#   Part F: Pairplot & Jointplot
#   Part G: FacetGrid — Multiple Subplots by Category
#   Part H: Regression Plots
#   Part I: Practical Examples
# =============================================================================

import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

def setup():
    os.makedirs('plots/deep_dive', exist_ok=True)

def get_data():
    """Load the Titanic and Tips datasets (both built into Seaborn)."""
    titanic = sns.load_dataset('titanic')
    tips = sns.load_dataset('tips')
    return titanic, tips


def part_a_themes():
    """Part A: Themes & Styles."""
    print("=" * 60)
    print("PART A: Themes & Styles")
    print("=" * 60)
    print("""
    Seaborn has 5 built-in themes:
      sns.set_theme(style='darkgrid')    ← default, good for most
      sns.set_theme(style='whitegrid')   ← clean white with gridlines
      sns.set_theme(style='dark')        ← dark without grid
      sns.set_theme(style='white')       ← clean white
      sns.set_theme(style='ticks')       ← white with tick marks

    Color palettes:
      'deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind'
      'viridis', 'plasma', 'coolwarm', 'Set1', 'Set2', 'husl'

    Context (scaling):
      sns.set_context('paper')     ← smallest
      sns.set_context('notebook')  ← default
      sns.set_context('talk')      ← presentations
      sns.set_context('poster')    ← largest
    """)
    sns.set_theme(style='whitegrid', palette='Set2')


def part_b_distribution(tips):
    """Part B: Distribution Plots."""
    print("=" * 60)
    print("PART B: Distribution Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. histplot — histogram with optional KDE
    sns.histplot(data=tips, x='total_bill', bins=20, kde=True, color='steelblue', ax=axes[0, 0])
    axes[0, 0].set_title('histplot (histogram + KDE)')

    # 2. kdeplot — smooth density curve
    sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, alpha=0.5, ax=axes[0, 1])
    axes[0, 1].set_title('kdeplot (by Lunch/Dinner)')

    # 3. histplot with hue (stacked)
    sns.histplot(data=tips, x='total_bill', hue='sex', multiple='stack', ax=axes[1, 0])
    axes[1, 0].set_title('histplot stacked by sex')

    # 4. ecdfplot — empirical cumulative distribution
    sns.ecdfplot(data=tips, x='total_bill', hue='time', ax=axes[1, 1])
    axes[1, 1].set_title('ecdfplot (cumulative distribution)')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/sns_01_distributions.png')
    plt.close()
    print("Saved 'plots/deep_dive/sns_01_distributions.png'\n")


def part_c_categorical(tips, titanic):
    """Part C: Categorical Plots."""
    print("=" * 60)
    print("PART C: Categorical Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. countplot — count of categories
    sns.countplot(data=titanic, x='pclass', hue='survived', palette='Set1', ax=axes[0, 0])
    axes[0, 0].set_title('countplot')

    # 2. barplot — mean (or other estimator) by category
    sns.barplot(data=tips, x='day', y='total_bill', hue='sex', palette='muted', ax=axes[0, 1])
    axes[0, 1].set_title('barplot (mean by day)')

    # 3. boxplot — distribution by quartiles
    sns.boxplot(data=tips, x='day', y='total_bill', hue='day',
                palette='pastel', legend=False, ax=axes[0, 2])
    axes[0, 2].set_title('boxplot')

    # 4. violinplot — density + boxplot combined
    sns.violinplot(data=tips, x='day', y='total_bill', hue='day',
                   palette='Set2', legend=False, ax=axes[1, 0])
    axes[1, 0].set_title('violinplot')

    # 5. swarmplot — individual data points (no overlap)
    sns.swarmplot(data=tips.sample(50, random_state=42), x='day', y='total_bill',
                  hue='day', palette='Set1', legend=False, ax=axes[1, 1], size=4)
    axes[1, 1].set_title('swarmplot')

    # 6. stripplot — individual points (with jitter)
    sns.stripplot(data=tips, x='day', y='total_bill', hue='day',
                  palette='dark', legend=False, alpha=0.5, ax=axes[1, 2])
    axes[1, 2].set_title('stripplot')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/sns_02_categorical.png')
    plt.close()
    print("Saved 'plots/deep_dive/sns_02_categorical.png'\n")


def part_d_relational(tips):
    """Part D: Relational Plots."""
    print("=" * 60)
    print("PART D: Relational Plots")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scatterplot with size + hue encoding
    sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time',
                    size='size', sizes=(20, 200), alpha=0.7, ax=axes[0])
    axes[0].set_title('scatterplot (size + hue)')

    # lineplot (great for time series)
    months = pd.DataFrame({
        'month': list(range(1, 13)) * 2,
        'sales': np.concatenate([
            np.array([100, 120, 130, 125, 140, 160, 155, 170, 180, 190, 210, 230]),
            np.array([80, 95, 110, 105, 115, 130, 125, 140, 155, 165, 180, 200])
        ]),
        'store': ['A']*12 + ['B']*12
    })
    sns.lineplot(data=months, x='month', y='sales', hue='store',
                 style='store', markers=True, ax=axes[1])
    axes[1].set_title('lineplot (time series)')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/sns_03_relational.png')
    plt.close()
    print("Saved 'plots/deep_dive/sns_03_relational.png'\n")


def part_e_heatmap():
    """Part E: Heatmaps."""
    print("=" * 60)
    print("PART E: Heatmaps")
    print("=" * 60)

    tips = sns.load_dataset('tips')
    numeric_cols = tips.select_dtypes(include='number')
    corr = numeric_cols.corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Basic correlation heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, ax=axes[0])
    axes[0].set_title('Correlation Heatmap')

    # Custom heatmap (pivot data)
    pivot = tips.pivot_table(values='tip', index='day', columns='time', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu',
                linewidths=1, ax=axes[1])
    axes[1].set_title('Average Tip by Day & Time')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/sns_04_heatmap.png')
    plt.close()
    print("Saved 'plots/deep_dive/sns_04_heatmap.png'\n")


def part_f_pair_joint(tips):
    """Part F: Pairplot & Jointplot."""
    print("=" * 60)
    print("PART F: Pairplot & Jointplot")
    print("=" * 60)

    # Pairplot — scatter matrix of all numeric columns
    g = sns.pairplot(tips, hue='time', palette='Set1', height=2, corner=True)
    g.fig.suptitle('Pairplot', y=1.02)
    g.savefig('plots/deep_dive/sns_05_pairplot.png')
    plt.close()
    print("Saved 'plots/deep_dive/sns_05_pairplot.png'")

    # Jointplot — bivariate + marginal distributions
    g = sns.jointplot(data=tips, x='total_bill', y='tip', hue='time',
                      kind='scatter', height=6)
    g.fig.suptitle('Jointplot', y=1.02)
    g.savefig('plots/deep_dive/sns_06_jointplot.png')
    plt.close()
    print("Saved 'plots/deep_dive/sns_06_jointplot.png'\n")


def part_g_facetgrid(tips):
    """Part G: FacetGrid — Multiple Subplots by Category."""
    print("=" * 60)
    print("PART G: FacetGrid")
    print("=" * 60)

    # catplot — categorical + facets (wraps countplot, barplot, boxplot, etc.)
    g = sns.catplot(data=tips, x='day', y='total_bill', hue='sex',
                    col='time', kind='box', height=4, aspect=1.2)
    g.fig.suptitle('Boxplot by Day, split by Lunch/Dinner', y=1.02)
    g.savefig('plots/deep_dive/sns_07_facetgrid.png')
    plt.close()

    # displot — distribution + facets
    g = sns.displot(data=tips, x='total_bill', hue='sex',
                    col='time', kind='hist', kde=True, height=4, aspect=1.2)
    g.savefig('plots/deep_dive/sns_08_displot_facet.png')
    plt.close()

    print("Saved 'plots/deep_dive/sns_07_facetgrid.png'")
    print("Saved 'plots/deep_dive/sns_08_displot_facet.png'\n")


def part_h_regression(tips):
    """Part H: Regression Plots."""
    print("=" * 60)
    print("PART H: Regression Plots")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # lmplot equivalent on an axes
    sns.regplot(data=tips, x='total_bill', y='tip', scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'}, ax=axes[0])
    axes[0].set_title('regplot (linear regression)')

    # residplot — residuals of a linear regression
    sns.residplot(data=tips, x='total_bill', y='tip', lowess=True,
                  scatter_kws={'alpha': 0.5}, ax=axes[1])
    axes[1].set_title('residplot (check linearity)')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/sns_09_regression.png')
    plt.close()
    print("Saved 'plots/deep_dive/sns_09_regression.png'\n")


def part_i_practical(titanic):
    """Part I: Practical — Complete EDA visualization dashboard."""
    print("=" * 60)
    print("PART I: Practical EDA Dashboard")
    print("=" * 60)

    titanic = titanic.copy()
    titanic['age'] = titanic['age'].fillna(titanic['age'].median())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Survival by class & gender
    sns.barplot(data=titanic, x='pclass', y='survived', hue='sex',
                palette='muted', ax=axes[0, 0])
    axes[0, 0].set_title('Survival Rate by Class & Gender')

    # Age distribution by survival
    sns.kdeplot(data=titanic, x='age', hue='survived', fill=True, alpha=0.4,
                palette={0: '#FF5722', 1: '#4CAF50'}, ax=axes[0, 1])
    axes[0, 1].set_title('Age Distribution by Survival')

    # Fare distribution
    sns.boxplot(data=titanic, x='pclass', y='fare', hue='pclass',
                palette='pastel', legend=False, ax=axes[1, 0])
    axes[1, 0].set_title('Fare by Passenger Class')
    axes[1, 0].set_ylim(0, 200)

    # Embarkation port counts
    sns.countplot(data=titanic, x='embarked', hue='survived',
                  palette='Set2', ax=axes[1, 1])
    axes[1, 1].set_title('Survival by Embarkation Port')

    fig.suptitle('Titanic EDA Dashboard', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('plots/deep_dive/sns_10_dashboard.png', bbox_inches='tight')
    plt.close()
    print("Saved 'plots/deep_dive/sns_10_dashboard.png'\n")


def main():
    setup()
    titanic, tips = get_data()

    part_a_themes()
    part_b_distribution(tips)
    part_c_categorical(tips, titanic)
    part_d_relational(tips)
    part_e_heatmap()
    part_f_pair_joint(tips)
    part_g_facetgrid(tips)
    part_h_regression(tips)
    part_i_practical(titanic)

    print("=" * 60)
    print("🎉 Seaborn Deep Dive Complete!")
    print("You now know the full Seaborn toolkit for data science.")
    print("=" * 60)

if __name__ == "__main__":
    main()
