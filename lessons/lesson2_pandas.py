# %% [markdown]
# # Lesson 2: Data Manipulation with Pandas 🐼
# Pandas is the most popular Python library for data manipulation and analysis.
# It introduces two primary data structures:
# 1. **Series**: A 1D labeled array (like a single column in a table).
# 2. **DataFrame**: A 2D labeled tabular data structure (like a database table or Excel sheet).
#
# Let's explore:
# 1. Creating Series and DataFrames
# 2. Reading CSV Data
# 3. Inspecting and Describing Data
# 4. Selection and Indexing (loc/iloc)
# 5. Filtering Data
# 6. Grouping and Aggregating Data
# 7. Exercises for you!

# %%
import os
import pandas as pd

def main():
    print("=== Pandas Data Manipulation Lesson ===")
    
    # Get correct path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'sample_sales.csv')

# %% [markdown]
    # -------------------------------------------------------------------------
    # 1. Creating Series and DataFrames
    # -------------------------------------------------------------------------
# %%
    print("\n--- 1. Creating Series & DataFrames ---")
    # A Series: list of values with index labels
    s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
    print("Pandas Series:\n", s)
    
    # A DataFrame from a dictionary
    student_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [23, 25, 22, 24],
        'Grade': ['A', 'B', 'A', 'C']
    }
    df_students = pd.DataFrame(student_data)
    print("\nDataFrame from Dictionary:\n", df_students)

# %% [markdown]
    # -------------------------------------------------------------------------
    # 2. Reading CSV Data
    # -------------------------------------------------------------------------
# %%
    print("\n--- 2. Reading CSV Data ---")
    # Read our sample sales CSV file
    df_sales = pd.read_csv(csv_path)
    print("Loaded CSV into DataFrame successfully!")

# %% [markdown]
    # -------------------------------------------------------------------------
    # 3. Inspecting and Describing Data
    # -------------------------------------------------------------------------
# %%
    print("\n--- 3. Inspecting Data ---")
    print("df_sales Shape (rows, columns):", df_sales.shape)
    print("\n--- Head (first 3 rows) ---")
    print(df_sales.head(3))
    print("\n--- Info (columns, data types, nulls) ---")
    df_sales.info()
    print("\n--- Describe (summary statistics) ---")
    print(df_sales.describe())

# %% [markdown]
    # -------------------------------------------------------------------------
    # 4. Selection and Indexing (loc / iloc)
    # -------------------------------------------------------------------------
# %%
    print("\n--- 4. Selection and Indexing ---")
    # Bracket selection for columns
    print("Select a single column ('Product'):\n", df_sales['Product'].head(3))
    print("\nSelect multiple columns:\n", df_sales[['Product', 'Price']].head(3))
    
    # .loc[] (Label-based indexing: df.loc[row_labels, col_labels])
    # .iloc[] (Integer-based indexing: df.iloc[row_positions, col_positions])
    print("\nGet row at index 0 using .loc:\n", df_sales.loc[0])
    print("\nGet rows 0 to 2, columns 'Product' and 'Price' using .loc:\n", df_sales.loc[0:2, ['Product', 'Price']])
    print("\nGet cell at row 1, col 2 using .iloc:\n", df_sales.iloc[1, 2])

# %% [markdown]
    # -------------------------------------------------------------------------
    # 5. Filtering Data
    # -------------------------------------------------------------------------
# %%
    print("\n--- 5. Filtering Data ---")
    # Filter sales where Price is greater than 200
    expensive_sales = df_sales[df_sales['Price'] > 200]
    print("Sales where Price > 200:\n", expensive_sales)
    
    # Filter using multiple conditions (& = AND, | = OR)
    us_electronics = df_sales[(df_sales['Country'] == 'US') & (df_sales['Category'] == 'Electronics')]
    print("\nUS Electronics sales:\n", us_electronics)

# %% [markdown]
    # -------------------------------------------------------------------------
    # 6. Grouping and Aggregating Data
    # -------------------------------------------------------------------------
# %%
    print("\n--- 6. Grouping & Aggregating ---")
    # Calculate revenue (Quantity * Price) and add it as a new column
    df_sales['Revenue'] = df_sales['Quantity'] * df_sales['Price']
    print("Added Revenue column. Head:\n", df_sales[['Product', 'Revenue']].head(3))
    
    # Group by Category and compute sum of Revenue & mean of Price
    category_summary = df_sales.groupby('Category').agg({
        'Revenue': 'sum',
        'Quantity': 'sum',
        'Price': 'mean'
    })
    print("\nSummary by Category:\n", category_summary)
    
    # Value counts of a column
    print("\nSales count by Country:\n", df_sales['Country'].value_counts())

    # -------------------------------------------------------------------------
    # YOUR TURN! (Exercises)
# %% [markdown]
    # Run the script first, then write your code in the functions below.
    # -------------------------------------------------------------------------
# %%
    print("\n==============================================")
    print("Now run the exercises below by calling them in main()!")
    print("==============================================")
    
    run_exercises(df_sales)

def run_exercises(df):
    print("\n--- Running Exercises ---")
    
    # Exercise 1: Calculate the total revenue of all orders combined.
    # Hint: Use df['Revenue'].sum()
    # TODO: Write your code here
    total_revenue = None
    print("Exercise 1 Total Revenue:", total_revenue)
    
    # Exercise 2: Filter the dataframe to show only sales in the 'UK'.
    # TODO: Write your code here
    uk_sales = None
    print("\nExercise 2 UK Sales:\n", uk_sales)
    
    # Exercise 3: Group by 'Product' and find the total 'Quantity' sold for each product.
    # Hint: Use df.groupby('Product')['Quantity'].sum()
    # TODO: Write your code here
    product_quantities = None
    print("\nExercise 3 Total Quantity per Product:\n", product_quantities)

if __name__ == "__main__":
    main()
