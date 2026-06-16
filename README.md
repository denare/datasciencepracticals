# Data Science Practicals 🚀

Welcome! This workspace is ready for you to start learning and practicing Data Science.

## Setup & Environment Details

A virtual environment has been created and configured for you at `.venv/` using Python 3.14.
All necessary data science libraries are installed, including:
- **NumPy** (`numpy`): For efficient numerical computation and array manipulation.
- **Pandas** (`pandas`): For data analysis, manipulation, and structured DataFrame structures.
- **Matplotlib** (`matplotlib`): For creating static, animated, and interactive visualizations.
- **Seaborn** (`seaborn`): High-level interface built on top of Matplotlib for drawing beautiful statistical graphics.
- **Scikit-learn** (`scikit-learn`): The gold standard library for machine learning algorithms, preprocessing, and model evaluation.
- **Jupyter** (`jupyter`): For interactive notebook environments where you can write code, see plots, and write documentation in one place.

---

## How to Get Started

### 1. Activate the Virtual Environment

Before running any script or notebook, activate the virtual environment in your terminal:

```bash
source .venv/bin/activate
```

*(You will see `(.venv)` in your terminal prompt, indicating it is active. To deactivate later, simply type `deactivate`.)*

### 2. Run the Starter Script

We have provided a starter script (`data_science_intro.py`) that loads the famous Iris dataset, performs basic data exploration with Pandas and NumPy, generates two plots (saved to the `plots/` folder), and trains a Random Forest Classifier with Scikit-learn.

Run it with:
```bash
python data_science_intro.py
```

Check out the outputs in your terminal and open the generated images in `plots/sepal_scatter.png` and `plots/correlation_heatmap.png` to see the visualizations.

### 3. Launch Jupyter Notebook / Lab

Since you are learning data science, writing code interactively in Jupyter Notebooks is highly recommended. To launch it, run:

```bash
jupyter notebook
# OR
jupyter lab
```

This will start a local server and print a link to open in your web browser. You can create a new notebook (using the Python 3 kernel in `.venv`) and start experimenting!

---

## Basic Concepts to Learn First

1. **NumPy Arrays:** Learn how to create arrays (`np.array`), use slicing, reshaping, and mathematical operations.
2. **Pandas DataFrames:** Learn how to read datasets (e.g., `pd.read_csv`), filter rows/columns, handle missing values, and group data (`df.groupby`).
3. **Data Visualization:** Learn how to plot distributions (histograms), correlations (scatter plots, heatmaps), and categorical variables (bar charts, box plots).
4. **Machine Learning Workflow:** Learn how to prepare features, split data (`train_test_split`), select a model, train it (`model.fit`), and evaluate performance (`model.predict` and metrics).

Happy Learning! 📊🐍
