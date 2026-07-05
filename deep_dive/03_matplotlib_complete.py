# =============================================================================
# DEEP DIVE 3: Matplotlib Complete Reference 📈
# =============================================================================
# Matplotlib is the foundational plotting library in Python.
# It gives you full control over every element of a figure.
#
# This script covers EVERYTHING you need:
#   Part A: Figure & Axes Architecture
#   Part B: Line Plots
#   Part C: Scatter Plots
#   Part D: Bar Charts
#   Part E: Histograms
#   Part F: Pie Charts
#   Part G: Subplots & Layouts
#   Part H: Customization (colors, styles, annotations)
#   Part I: 3D Plots
#   Part J: Saving Figures
# =============================================================================

import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def setup():
    os.makedirs('plots/deep_dive', exist_ok=True)

def part_a_architecture():
    """Part A: Figure & Axes Architecture."""
    print("=" * 60)
    print("PART A: Figure & Axes Architecture")
    print("=" * 60)
    print("""
    KEY CONCEPT: Matplotlib has two layers:
    ┌─────────────────────────────────┐
    │  Figure (the canvas/window)     │
    │  ┌───────────┐ ┌───────────┐   │
    │  │  Axes 1   │ │  Axes 2   │   │
    │  │ (a plot)  │ │ (a plot)  │   │
    │  └───────────┘ └───────────┘   │
    └─────────────────────────────────┘

    - Figure = entire image / canvas
    - Axes   = individual plot area (has x-axis, y-axis, title, etc.)
    - Use fig, ax = plt.subplots() for best practice
    """)

    # Recommended approach
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title('Basic Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_01_basic.png')
    plt.close()
    print("Saved 'plots/deep_dive/mpl_01_basic.png'\n")


def part_b_line():
    """Part B: Line Plots."""
    print("=" * 60)
    print("PART B: Line Plots")
    print("=" * 60)

    x = np.linspace(0, 2 * np.pi, 100)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Multiple lines with different styles
    ax.plot(x, np.sin(x), label='sin(x)', color='#2196F3', linewidth=2)
    ax.plot(x, np.cos(x), label='cos(x)', color='#FF5722', linewidth=2, linestyle='--')
    ax.plot(x, np.sin(x) * np.cos(x), label='sin·cos', color='#4CAF50',
            linewidth=1.5, linestyle='-.', marker='o', markevery=10, markersize=5)

    ax.set_title('Trigonometric Functions', fontsize=14, fontweight='bold')
    ax.set_xlabel('x (radians)')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)  # Horizontal line at y=0

    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_02_line.png')
    plt.close()
    print("Saved 'plots/deep_dive/mpl_02_line.png'\n")


def part_c_scatter():
    """Part C: Scatter Plots."""
    print("=" * 60)
    print("PART C: Scatter Plots")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)
    sizes = rng.uniform(20, 200, n)
    colors = y

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x, y, c=colors, s=sizes, cmap='plasma', alpha=0.7, edgecolors='white')
    fig.colorbar(scatter, ax=ax, label='Y Value')
    ax.set_title('Scatter Plot with Size & Color Encoding')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_03_scatter.png')
    plt.close()
    print("Saved 'plots/deep_dive/mpl_03_scatter.png'\n")


def part_d_bar():
    """Part D: Bar Charts."""
    print("=" * 60)
    print("PART D: Bar Charts")
    print("=" * 60)

    categories = ['Python', 'JavaScript', 'Java', 'C++', 'Go']
    values = [85, 78, 65, 55, 45]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Vertical bar
    bars = axes[0].bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Language Popularity (Vertical)')
    axes[0].set_ylabel('Score')
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(val), ha='center', fontweight='bold')

    # Horizontal bar
    axes[1].barh(categories, values, color=colors, edgecolor='white', linewidth=1.5)
    axes[1].set_title('Language Popularity (Horizontal)')
    axes[1].set_xlabel('Score')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_04_bar.png')
    plt.close()
    print("Saved 'plots/deep_dive/mpl_04_bar.png'\n")


def part_e_histogram():
    """Part E: Histograms."""
    print("=" * 60)
    print("PART E: Histograms")
    print("=" * 60)

    rng = np.random.default_rng(42)
    data1 = rng.normal(50, 10, 1000)
    data2 = rng.normal(60, 15, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Basic histogram
    axes[0].hist(data1, bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
    axes[0].set_title('Single Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(data1.mean(), color='red', linestyle='--', label=f'Mean: {data1.mean():.1f}')
    axes[0].legend()

    # Overlapping histograms
    axes[1].hist(data1, bins=30, alpha=0.5, color='#2196F3', label='Group A')
    axes[1].hist(data2, bins=30, alpha=0.5, color='#FF5722', label='Group B')
    axes[1].set_title('Overlapping Distributions')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_05_histogram.png')
    plt.close()
    print("Saved 'plots/deep_dive/mpl_05_histogram.png'\n")


def part_f_pie():
    """Part F: Pie Charts."""
    print("=" * 60)
    print("PART F: Pie Charts")
    print("=" * 60)

    labels = ['Python', 'JavaScript', 'Java', 'Others']
    sizes = [40, 30, 15, 15]
    explode = (0.05, 0, 0, 0)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9E9E9E']

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', startangle=90, shadow=True,
           textprops={'fontsize': 12})
    ax.set_title('Programming Language Usage', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_06_pie.png')
    plt.close()
    print("Saved 'plots/deep_dive/mpl_06_pie.png'\n")


def part_g_subplots():
    """Part G: Subplots & Layouts."""
    print("=" * 60)
    print("PART G: Subplots & Layouts")
    print("=" * 60)

    # Regular grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x = np.linspace(0, 10, 50)

    axes[0, 0].plot(x, np.sin(x), 'b-')
    axes[0, 0].set_title('sin(x)')

    axes[0, 1].plot(x, np.cos(x), 'r--')
    axes[0, 1].set_title('cos(x)')

    axes[1, 0].plot(x, np.exp(-x/5), 'g-.')
    axes[1, 0].set_title('exp(-x/5)')

    axes[1, 1].plot(x, np.log(x + 1), 'm:')
    axes[1, 1].set_title('log(x+1)')

    fig.suptitle('2x2 Subplot Grid', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_07_subplots.png', bbox_inches='tight')
    plt.close()

    # Custom layout with GridSpec
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, :2])   # Top-left spanning 2 columns
    ax2 = fig.add_subplot(gs[0, 2])    # Top-right
    ax3 = fig.add_subplot(gs[1, :])    # Bottom spanning all 3

    ax1.set_title('Wide Top-Left')
    ax2.set_title('Top-Right')
    ax3.set_title('Full-Width Bottom')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_08_gridspec.png')
    plt.close()
    print("Saved 'plots/deep_dive/mpl_07_subplots.png'")
    print("Saved 'plots/deep_dive/mpl_08_gridspec.png'\n")


def part_h_customization():
    """Part H: Customization — colors, styles, annotations."""
    print("=" * 60)
    print("PART H: Customization")
    print("=" * 60)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color='#2196F3', linewidth=2)

    # Annotations
    peak_x = np.pi / 2
    ax.annotate('Peak', xy=(peak_x, 1), xytext=(peak_x + 1.5, 1.2),
                fontsize=12, arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Fill between
    ax.fill_between(x, y, alpha=0.15, color='#2196F3')

    # Shaded region
    ax.axvspan(4, 6, alpha=0.1, color='red', label='Region of Interest')

    # Custom ticks
    ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
    ax.set_xticklabels(['0', 'π', '2π', '3π'])

    # Styling
    ax.set_title('Customized Plot', fontsize=16, fontweight='bold', color='#333')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('sin(x)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_09_custom.png', dpi=150)
    plt.close()
    print("Saved 'plots/deep_dive/mpl_09_custom.png'\n")


def part_i_3d():
    """Part I: 3D Plots."""
    print("=" * 60)
    print("PART I: 3D Plots")
    print("=" * 60)

    fig = plt.figure(figsize=(10, 5))

    # 3D scatter
    ax1 = fig.add_subplot(121, projection='3d')
    rng = np.random.default_rng(42)
    x, y, z = rng.normal(0, 1, 100), rng.normal(0, 1, 100), rng.normal(0, 1, 100)
    ax1.scatter(x, y, z, c=z, cmap='viridis', s=30)
    ax1.set_title('3D Scatter')

    # 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    X = np.linspace(-3, 3, 50)
    Y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    ax2.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
    ax2.set_title('3D Surface')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/mpl_10_3d.png')
    plt.close()
    print("Saved 'plots/deep_dive/mpl_10_3d.png'\n")


def part_j_saving():
    """Part J: Saving Figures."""
    print("=" * 60)
    print("PART J: Saving Figures")
    print("=" * 60)
    print("""
    plt.savefig('file.png')            # PNG (raster, default)
    plt.savefig('file.pdf')            # PDF (vector, for papers)
    plt.savefig('file.svg')            # SVG (vector, for web)
    plt.savefig('file.png', dpi=300)   # High resolution
    plt.savefig('file.png',
                bbox_inches='tight',   # Remove extra whitespace
                transparent=True)      # Transparent background
    """)


def main():
    setup()
    part_a_architecture()
    part_b_line()
    part_c_scatter()
    part_d_bar()
    part_e_histogram()
    part_f_pie()
    part_g_subplots()
    part_h_customization()
    part_i_3d()
    part_j_saving()

    print("=" * 60)
    print("🎉 Matplotlib Deep Dive Complete!")
    print("You now know the full Matplotlib toolkit for data science.")
    print("=" * 60)

if __name__ == "__main__":
    main()
