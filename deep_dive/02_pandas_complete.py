# =============================================================================
# %% [markdown]
# DEEP DIVE 2: Pandas Complete Reference 🐼
# =============================================================================
# Pandas is the primary library for data manipulation and analysis in Python.
# It provides two core data structures: Series (1D) and DataFrame (2D).
#
# This script covers EVERYTHING you need:
#   Part A: Series — Creation, Indexing, Operations
#   Part B: DataFrame — Creation & Loading Data
#   Part C: Inspecting & Describing Data
#   Part D: Selection — loc, iloc, Bracket Notation
#   Part E: Filtering & Boolean Indexing
#   Part F: Adding, Removing & Renaming Columns
#   Part G: Handling Missing Data
#   Part H: GroupBy & Aggregation
#   Part I: Merging, Joining & Concatenation
#   Part J: Pivot Tables & Crosstabs
#   Part K: String Operations
#   Part L: Date & Time Operations
#   Part M: Apply, Map & Lambda
#   Part N: Sorting & Ranking
#   Part O: Practical Examples
# =============================================================================

# %%
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd

def part_a_series():
    """Part A: Series — 1D labeled array."""
    print("=" * 60)
    print("PART A: Series")
    print("=" * 60)

    # Creation
    s1 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
    print("Series with custom index:\n", s1)

    s2 = pd.Series({'x': 100, 'y': 200, 'z': 300})
    print("\nSeries from dict:\n", s2)

    # Indexing
    print("\ns1['b']    :", s1['b'])
    print("s1[['a','c']]:", s1[['a', 'c']].values)

    # Operations (element-wise)
    print("\ns1 * 2:\n", s1 * 2)
    print("\ns1 > 20:\n", s1[s1 > 20])

    # Useful methods
    print("\n.sum()  :", s1.sum())
    print(".mean() :", s1.mean())
    print(".max()  :", s1.max())
    print(".idxmax():", s1.idxmax(), "(label of max value)\n")


def part_b_dataframe_creation():
    """Part B: DataFrame — Creation & Loading Data."""
    print("=" * 60)
    print("PART B: DataFrame Creation & Loading")
    print("=" * 60)

    # 1. From a dictionary
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Age': [25, 30, 35, 28],
        'City': ['Nairobi', 'Lagos', 'Accra', 'Nairobi'],
        'Salary': [50000, 60000, 70000, 55000]
    }
    df = pd.DataFrame(data)
    print("1. From dict:\n", df)

    # 2. From a list of lists
    df2 = pd.DataFrame(
        [[1, 'X'], [2, 'Y'], [3, 'Z']],
        columns=['ID', 'Label']
    )
    print("\n2. From list of lists:\n", df2)

    # 3. From a CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lessons_dir = os.path.join(os.path.dirname(current_dir), 'lessons')
    csv_path = os.path.join(lessons_dir, 'sample_sales.csv')
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        print("\n3. From CSV (first 3 rows):\n", df_csv.head(3))

    # 4. From NumPy array
    arr = np.random.randint(1, 100, size=(3, 3))
    df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
    print("\n4. From NumPy array:\n", df3, "\n")

    return df


def part_c_inspect(df):
    """Part C: Inspecting & Describing Data."""
    print("=" * 60)
    print("PART C: Inspecting & Describing Data")
    print("=" * 60)

    print(".head(2):\n", df.head(2))
    print("\n.tail(2):\n", df.tail(2))
    print("\n.shape  :", df.shape)
    print(".columns:", list(df.columns))
    print(".dtypes:\n", df.dtypes)
    print("\n.describe():\n", df.describe())
    print("\n.info():")
    df.info()
    print("\n.nunique():\n", df.nunique())
    print("\n.value_counts('City'):\n", df['City'].value_counts(), "\n")


def part_d_selection(df):
    """Part D: Selection — loc, iloc, Bracket Notation."""
    print("=" * 60)
    print("PART D: Selection (loc / iloc / brackets)")
    print("=" * 60)

    # Bracket notation
    print("Single column (Series):\n", df['Name'])
    print("\nMultiple columns:\n", df[['Name', 'Salary']])

    # .loc — label-based (inclusive on both sides)
    print("\n.loc[0] (first row):\n", df.loc[0])
    print("\n.loc[0:2, 'Name':'City']:\n", df.loc[0:2, 'Name':'City'])

    # .iloc — integer position-based (exclusive on end)
    print("\n.iloc[0] (first row):\n", df.iloc[0])
    print("\n.iloc[0:2, 0:2]:\n", df.iloc[0:2, 0:2])

    # .at / .iat — fast scalar access
    print("\n.at[1, 'Name']:", df.at[1, 'Name'])
    print(".iat[1, 0]    :", df.iat[1, 0], "\n")


def part_e_filtering(df):
    """Part E: Filtering & Boolean Indexing."""
    print("=" * 60)
    print("PART E: Filtering & Boolean Indexing")
    print("=" * 60)

    # Single condition
    high_salary = df[df['Salary'] > 55000]
    print("Salary > 55000:\n", high_salary)

    # Multiple conditions (& = AND, | = OR, ~ = NOT)
    nairobi_high = df[(df['City'] == 'Nairobi') & (df['Salary'] > 50000)]
    print("\nNairobi AND Salary > 50000:\n", nairobi_high)

    # .isin() — check membership
    cities_filter = df[df['City'].isin(['Nairobi', 'Accra'])]
    print("\nCity in ['Nairobi', 'Accra']:\n", cities_filter)

    # .between()
    age_range = df[df['Age'].between(26, 32)]
    print("\nAge between 26 and 32:\n", age_range)

    # .query() — SQL-like string filtering
    result = df.query("Age > 27 and City == 'Nairobi'")
    print("\n.query('Age > 27 and City == Nairobi'):\n", result, "\n")


def part_f_columns(df):
    """Part F: Adding, Removing & Renaming Columns."""
    print("=" * 60)
    print("PART F: Adding, Removing & Renaming Columns")
    print("=" * 60)

    df = df.copy()

    # Add a new column
    df['Bonus'] = df['Salary'] * 0.1
    print("Added 'Bonus':\n", df)

    # Add conditional column
    df['Senior'] = df['Age'].apply(lambda x: 'Yes' if x >= 30 else 'No')
    print("\nAdded 'Senior':\n", df)

    # Rename columns
    df_renamed = df.rename(columns={'Name': 'FullName', 'City': 'Location'})
    print("\nRenamed columns:", list(df_renamed.columns))

    # Drop columns
    df_dropped = df.drop(columns=['Bonus', 'Senior'])
    print("After drop:", list(df_dropped.columns), "\n")


def part_g_missing():
    """Part G: Handling Missing Data."""
    print("=" * 60)
    print("PART G: Handling Missing Data")
    print("=" * 60)

    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 2, 3, np.nan],
        'C': [1, 2, 3, 4]
    })
    print("Data with NaN:\n", df)

    # Detect missing values
    print("\n.isnull():\n", df.isnull())
    print("\n.isnull().sum():\n", df.isnull().sum())

    # Drop rows with any NaN
    print("\ndropna():\n", df.dropna())

    # Fill missing values
    print("fillna(0):\n", df.fillna(0))
    print("fillna(mean):\n", df.fillna(df.mean(numeric_only=True)))

    # Forward fill / backward fill
    print("ffill():\n", df.ffill())
    print("bfill():\n", df.bfill(), "\n")


def part_h_groupby(df):
    """Part H: GroupBy & Aggregation."""
    print("=" * 60)
    print("PART H: GroupBy & Aggregation")
    print("=" * 60)

    # Basic groupby
    grouped = df.groupby('City')['Salary'].mean()
    print("Mean salary by city:\n", grouped)

    # Multiple aggregations
    agg = df.groupby('City').agg({
        'Salary': ['mean', 'sum', 'count'],
        'Age': ['min', 'max']
    })
    print("\nMultiple agg by city:\n", agg)

    # .agg with named output
    named_agg = df.groupby('City').agg(
        avg_salary=('Salary', 'mean'),
        num_people=('Name', 'count')
    )
    print("\nNamed aggregation:\n", named_agg)

    # Transform — returns same-sized output (useful for adding group stats back)
    df_copy = df.copy()
    df_copy['city_avg_salary'] = df.groupby('City')['Salary'].transform('mean')
    print("\nTransform (city avg added):\n", df_copy, "\n")


def part_i_merge():
    """Part I: Merging, Joining & Concatenation."""
    print("=" * 60)
    print("PART I: Merging, Joining & Concatenation")
    print("=" * 60)

    # Sample DataFrames
    orders = pd.DataFrame({
        'order_id': [1, 2, 3, 4],
        'customer_id': [101, 102, 103, 104],
        'amount': [250, 150, 300, 200]
    })
    customers = pd.DataFrame({
        'customer_id': [101, 102, 103, 105],
        'name': ['Alice', 'Bob', 'Charlie', 'Eve']
    })

    # Inner merge (only matching keys)
    inner = pd.merge(orders, customers, on='customer_id', how='inner')
    print("Inner merge:\n", inner)

    # Left merge (keep all from left)
    left = pd.merge(orders, customers, on='customer_id', how='left')
    print("\nLeft merge:\n", left)

    # Outer merge (keep all)
    outer = pd.merge(orders, customers, on='customer_id', how='outer')
    print("\nOuter merge:\n", outer)

    # Concatenation (stacking DataFrames)
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    concat = pd.concat([df1, df2], ignore_index=True)
    print("\nConcat (vertical):\n", concat, "\n")


def part_j_pivot():
    """Part J: Pivot Tables & Crosstabs."""
    print("=" * 60)
    print("PART J: Pivot Tables & Crosstabs")
    print("=" * 60)

    sales = pd.DataFrame({
        'Date': ['Mon', 'Mon', 'Tue', 'Tue', 'Wed', 'Wed'],
        'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Revenue': [100, 200, 150, 250, 130, 220]
    })

    # Pivot table
    pivot = sales.pivot_table(values='Revenue', index='Date', columns='Product', aggfunc='sum')
    print("Pivot table:\n", pivot)

    # Crosstab
    data = pd.DataFrame({
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'Survived': [1, 1, 0, 1, 0, 0]
    })
    ct = pd.crosstab(data['Gender'], data['Survived'], margins=True)
    print("\nCrosstab:\n", ct, "\n")


def part_k_strings():
    """Part K: String Operations (.str accessor)."""
    print("=" * 60)
    print("PART K: String Operations")
    print("=" * 60)

    names = pd.Series(['  Alice Smith  ', 'BOB JONES', 'charlie brown', 'Diana_Prince'])
    print("Original:\n", names)
    print("\n.str.lower() :\n", names.str.lower())
    print("\n.str.upper() :\n", names.str.upper())
    print("\n.str.strip() :\n", names.str.strip())
    print("\n.str.title() :\n", names.str.title())
    print("\n.str.contains('o', case=False):\n", names.str.contains('o', case=False))
    print("\n.str.replace('_', ' '):\n", names.str.replace('_', ' '))
    print("\n.str.split():\n", names.str.strip().str.split())
    print("\n.str.len():\n", names.str.strip().str.len(), "\n")


def part_l_datetime():
    """Part L: Date & Time Operations."""
    print("=" * 60)
    print("PART L: Date & Time Operations")
    print("=" * 60)

    # Create datetime column
    dates = pd.to_datetime(['2026-01-15', '2026-03-22', '2026-06-10', '2026-12-01'])
    df = pd.DataFrame({'date': dates, 'sales': [100, 200, 150, 300]})
    print("DataFrame:\n", df)

    # Extract date components using .dt accessor
    print("\n.dt.year  :", df['date'].dt.year.values)
    print(".dt.month :", df['date'].dt.month.values)
    print(".dt.day   :", df['date'].dt.day.values)
    print(".dt.day_name():", df['date'].dt.day_name().values)

    # Date range
    date_range = pd.date_range(start='2026-01-01', periods=5, freq='W')
    print("\ndate_range (weekly):\n", date_range)

    # Resampling (time-based groupby)
    ts = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=10, freq='D'),
        'value': np.random.randint(10, 50, 10)
    }).set_index('date')
    weekly = ts.resample('W').sum()
    print("\nWeekly resample:\n", weekly, "\n")


def part_m_apply():
    """Part M: Apply, Map & Lambda."""
    print("=" * 60)
    print("PART M: Apply, Map & Lambda")
    print("=" * 60)

    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [85, 92, 78]
    })

    # .apply() on a column (Series) — element-wise with a function
    df['Grade'] = df['Score'].apply(lambda x: 'Pass' if x >= 80 else 'Fail')
    print("apply() with lambda:\n", df)

    # .apply() on entire DataFrame (axis=1 for row-wise)
    df['Summary'] = df.apply(
        lambda row: f"{row['Name']}: {row['Grade']}", axis=1
    )
    print("\napply() row-wise:\n", df)

    # .map() — for Series (element-wise mapping with dict)
    grade_map = {'Pass': '✅', 'Fail': '❌'}
    df['Icon'] = df['Grade'].map(grade_map)
    print("\nmap() with dict:\n", df)

    # .applymap() — element-wise on entire DataFrame (renamed to .map() in newer pandas)
    nums = pd.DataFrame({'A': [1.123, 2.456], 'B': [3.789, 4.012]})
    rounded = nums.map(lambda x: round(x, 1))
    print("\nDataFrame .map() (element-wise):\n", rounded, "\n")


def part_n_sorting():
    """Part N: Sorting & Ranking."""
    print("=" * 60)
    print("PART N: Sorting & Ranking")
    print("=" * 60)

    df = pd.DataFrame({
        'Name': ['Charlie', 'Alice', 'Bob', 'Diana'],
        'Score': [88, 95, 88, 72]
    })

    # Sort by values
    print("Sort by Score (asc):\n", df.sort_values('Score'))
    print("\nSort by Score (desc):\n", df.sort_values('Score', ascending=False))

    # Sort by multiple columns
    print("\nSort by Score then Name:\n",
          df.sort_values(['Score', 'Name'], ascending=[False, True]))

    # Ranking
    df['Rank'] = df['Score'].rank(ascending=False, method='min')
    print("\nRanking:\n", df)

    # nlargest / nsmallest
    print("\nTop 2 scores:\n", df.nlargest(2, 'Score'))
    print("\nBottom 2 scores:\n", df.nsmallest(2, 'Score'), "\n")


def part_o_practical():
    """Part O: Practical Examples."""
    print("=" * 60)
    print("PART O: Practical Examples")
    print("=" * 60)

    # 1. Read CSV, clean, and summarize
    print("1. Method chaining (fluent API):")
    df = pd.DataFrame({
        'name': ['  Alice ', 'BOB', ' charlie'],
        'score': [85, None, 92],
        'city': ['NY', 'LA', 'NY']
    })
    result = (df
              .assign(name=lambda x: x['name'].str.strip().str.title())
              .fillna({'score': df['score'].median()})
              .query('score > 80')
              )
    print(result)

    # 2. Binning continuous values
    print("\n2. Binning ages into categories:")
    ages = pd.Series([5, 15, 25, 35, 45, 55, 65, 75])
    bins = [0, 18, 35, 60, 100]
    labels = ['Child', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(ages, bins=bins, labels=labels)
    print(categories.value_counts())

    # 3. One-hot encoding with get_dummies
    print("\n3. One-hot encoding:")
    colors = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'blue']})
    dummies = pd.get_dummies(colors, prefix='color')
    print(dummies)

    print()


def main():
    part_a_series()
    df = part_b_dataframe_creation()
    part_c_inspect(df)
    part_d_selection(df)
    part_e_filtering(df)
    part_f_columns(df)
    part_g_missing()
    part_h_groupby(df)
    part_i_merge()
    part_j_pivot()
    part_k_strings()
    part_l_datetime()
    part_m_apply()
    part_n_sorting()
    part_o_practical()

    print("=" * 60)
    print("🎉 Pandas Deep Dive Complete!")
    print("You now know the full Pandas toolkit for data science.")
    print("=" * 60)

if __name__ == "__main__":
    main()
