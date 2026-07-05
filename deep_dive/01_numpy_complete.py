# =============================================================================
# %% [markdown]
# DEEP DIVE 1: NumPy Complete Reference 🧮
# =============================================================================
# NumPy is the backbone of scientific computing in Python. Every data science
# library (Pandas, Scikit-learn, TensorFlow) is built on top of it.
#
# This script covers EVERYTHING you need to know:
#   Part A: Array Creation (10 methods)
#   Part B: Array Attributes & Inspection
#   Part C: Reshaping & Manipulating
#   Part D: Indexing, Slicing & Fancy Indexing
#   Part E: Mathematical Operations
#   Part F: Broadcasting Rules
#   Part G: Aggregation & Statistical Functions
#   Part H: Linear Algebra
#   Part I: Random Number Generation
#   Part J: Sorting & Searching
#   Part K: Practical Examples
# =============================================================================

# pyrefly: ignore [missing-import]
# %%
import numpy as np

def part_a_creation():
    """Part A: Array Creation — 10 ways to create arrays."""
    print("=" * 60)
    print("PART A: Array Creation")
    print("=" * 60)

    # 1. From a Python list
    arr = np.array([1, 2, 3, 4, 5])
    print("1. From list:", arr)

    # 2. From nested lists (2D)
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("2. 2D from nested lists:\n", matrix)

    # 3. Zeros — all elements are 0
    zeros = np.zeros((3, 4))
    print("3. Zeros (3x4):\n", zeros)

    # 4. Ones — all elements are 1
    ones = np.ones((2, 3))
    print("4. Ones (2x3):\n", ones)

    # 5. Full — fill with any value
    full = np.full((2, 2), fill_value=7)
    print("5. Full (2x2, fill=7):\n", full)

    # 6. arange — like Python's range(), but returns an array
    arange = np.arange(0, 20, 3)  # start, stop (exclusive), step
    print("6. Arange(0, 20, 3):", arange)

    # 7. linspace — evenly spaced values between start and end (inclusive)
    linspace = np.linspace(0, 1, 5)  # start, stop, number of points
    print("7. Linspace(0, 1, 5):", linspace)

    # 8. eye — identity matrix (diagonal of 1s)
    identity = np.eye(3)
    print("8. Identity 3x3:\n", identity)

    # 9. empty — uninitialized (fast, but values are garbage)
    empty = np.empty((2, 2))
    print("9. Empty (2x2, uninitialized):\n", empty)

    # 10. From existing array (copy vs view)
    original = np.array([10, 20, 30])
    copy = original.copy()      # Independent copy
    view = original.view()      # Shares memory with original
    copy[0] = 999
    view[1] = 888
    print("10a. Original after view change:", original)  # [10, 888, 30]
    print("10b. Copy (independent):", copy)              # [999, 20, 30]
    print("    KEY INSIGHT: .copy() is independent, .view() shares memory!\n")


def part_b_attributes():
    """Part B: Array Attributes & Inspection."""
    print("=" * 60)
    print("PART B: Array Attributes & Inspection")
    print("=" * 60)

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    print("Array:\n", arr)
    print("  .shape  :", arr.shape)     # (2, 3) — rows, cols
    print("  .ndim   :", arr.ndim)      # 2 — number of dimensions
    print("  .size   :", arr.size)      # 6 — total elements
    print("  .dtype  :", arr.dtype)     # int64 — data type
    print("  .nbytes :", arr.nbytes)    # 48 — memory in bytes
    print("  .T      :\n", arr.T)      # Transpose

    # Specifying data types explicitly
    float_arr = np.array([1, 2, 3], dtype=np.float32)
    print("\n  float32 array:", float_arr, "| dtype:", float_arr.dtype)

    # Converting data types
    int_arr = float_arr.astype(np.int64)
    print("  Converted to int64:", int_arr, "| dtype:", int_arr.dtype, "\n")


def part_c_reshape():
    """Part C: Reshaping & Manipulating."""
    print("=" * 60)
    print("PART C: Reshaping & Manipulating")
    print("=" * 60)

    arr = np.arange(1, 13)  # [1, 2, 3, ..., 12]
    print("Original:", arr)

    # reshape — change shape without changing data
    reshaped = arr.reshape(3, 4)
    print("reshape(3, 4):\n", reshaped)

    reshaped2 = arr.reshape(2, -1)  # -1 means "figure it out automatically"
    print("reshape(2, -1):\n", reshaped2)

    # flatten — collapse to 1D (returns a copy)
    flat = reshaped.flatten()
    print("flatten():", flat)

    # ravel — collapse to 1D (returns a view when possible)
    raveled = reshaped.ravel()
    print("ravel():", raveled)

    # Stacking arrays
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    vstacked = np.vstack([a, b])  # Vertical stack (rows)
    hstacked = np.hstack([a, b])  # Horizontal stack
    print("vstack:\n", vstacked)
    print("hstack:", hstacked)

    # Splitting arrays
    big = np.arange(1, 13).reshape(3, 4)
    splits = np.hsplit(big, 2)  # Split into 2 along columns
    print("hsplit:\n", splits[0], "\n", splits[1])

    # Adding/removing dimensions
    arr1d = np.array([1, 2, 3])
    row_vec = arr1d[np.newaxis, :]   # Shape: (1, 3)
    col_vec = arr1d[:, np.newaxis]   # Shape: (3, 1)
    print("newaxis row:", row_vec.shape, row_vec)
    print("newaxis col:", col_vec.shape, "\n", col_vec, "\n")


def part_d_indexing():
    """Part D: Indexing, Slicing & Fancy Indexing."""
    print("=" * 60)
    print("PART D: Indexing, Slicing & Fancy Indexing")
    print("=" * 60)

    # 1D indexing
    arr = np.array([10, 20, 30, 40, 50])
    print("Array:", arr)
    print("  arr[0]  :", arr[0])     # First element
    print("  arr[-1] :", arr[-1])    # Last element
    print("  arr[1:4]:", arr[1:4])   # Slice (index 1, 2, 3)
    print("  arr[::2]:", arr[::2])   # Every other element

    # 2D indexing
    matrix = np.arange(1, 10).reshape(3, 3)
    print("\nMatrix:\n", matrix)
    print("  matrix[0, 1]   :", matrix[0, 1])       # Row 0, Col 1
    print("  matrix[0, :]   :", matrix[0, :])       # Entire row 0
    print("  matrix[:, -1]  :", matrix[:, -1])      # Last column
    print("  matrix[0:2, 1:]:\n", matrix[0:2, 1:])  # Submatrix

    # Fancy indexing — use arrays of indices
    arr = np.array([10, 20, 30, 40, 50])
    indices = np.array([0, 3, 4])
    print("\nFancy indexing arr[[0,3,4]]:", arr[indices])

    # Boolean indexing (masking) — extremely important!
    arr = np.array([15, 22, 8, 35, 42, 3])
    mask = arr > 20
    print("Boolean mask (arr > 20):", mask)
    print("Filtered values:", arr[mask])

    # np.where — conditional selection
    result = np.where(arr > 20, arr, 0)  # Keep if > 20, else replace with 0
    print("np.where(arr > 20, arr, 0):", result, "\n")


def part_e_math():
    """Part E: Mathematical Operations."""
    print("=" * 60)
    print("PART E: Mathematical Operations")
    print("=" * 60)

    a = np.array([10, 20, 30, 40])
    b = np.array([1, 2, 3, 4])

    # Element-wise arithmetic
    print("a + b  :", a + b)
    print("a - b  :", a - b)
    print("a * b  :", a * b)
    print("a / b  :", a / b)
    print("a // b :", a // b)   # Floor division
    print("a % b  :", a % b)    # Modulo
    print("a ** 2 :", a ** 2)   # Power

    # Universal functions (ufuncs) — fast element-wise operations
    arr = np.array([1, 4, 9, 16, 25])
    print("\nnp.sqrt()  :", np.sqrt(arr))
    print("np.exp()   :", np.exp(np.array([0, 1, 2])))
    print("np.log()   :", np.log(np.array([1, np.e, np.e**2])))
    print("np.abs()   :", np.abs(np.array([-3, -1, 0, 2, 5])))
    print("np.sin()   :", np.sin(np.array([0, np.pi/2, np.pi])))

    # Rounding
    decimals = np.array([1.234, 5.678, 9.012])
    print("\nnp.round(2) :", np.round(decimals, 2))
    print("np.floor()  :", np.floor(decimals))
    print("np.ceil()   :", np.ceil(decimals), "\n")


def part_f_broadcasting():
    """Part F: Broadcasting Rules — how NumPy handles different shapes."""
    print("=" * 60)
    print("PART F: Broadcasting Rules")
    print("=" * 60)

    # Rule: NumPy stretches the smaller array to match the larger one
    # when their shapes are compatible.

    # Scalar broadcast
    arr = np.array([1, 2, 3])
    print("arr + 10 :", arr + 10)  # 10 is broadcasted to [10, 10, 10]

    # 1D + 2D broadcast
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    row = np.array([10, 20, 30])
    print("matrix + row:\n", matrix + row)  # row is added to each row of matrix

    # Column broadcast (need to reshape to column vector)
    col = np.array([[100], [200]])  # Shape: (2, 1)
    print("matrix + col:\n", matrix + col)  # col is added to each column

    # Practical: Normalize columns to zero mean
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    col_means = data.mean(axis=0)     # Mean of each column
    normalized = data - col_means     # Broadcasting subtracts from each row
    print("\nOriginal data:\n", data)
    print("Column means:", col_means)
    print("Normalized (zero-mean):\n", normalized, "\n")


def part_g_aggregations():
    """Part G: Aggregation & Statistical Functions."""
    print("=" * 60)
    print("PART G: Aggregation & Statistical Functions")
    print("=" * 60)

    data = np.array([[10, 20, 30],
                     [40, 50, 60],
                     [70, 80, 90]])
    print("Data:\n", data)

    # Global aggregations (entire array)
    print("\n--- Global ---")
    print("  sum   :", np.sum(data))
    print("  mean  :", np.mean(data))
    print("  median:", np.median(data))
    print("  std   :", np.std(data))
    print("  var   :", np.var(data))
    print("  min   :", np.min(data))
    print("  max   :", np.max(data))

    # Axis-based aggregations
    # axis=0 → collapse rows (operate DOWN columns)
    # axis=1 → collapse columns (operate ACROSS rows)
    print("\n--- Along axis=0 (column-wise) ---")
    print("  sum   :", np.sum(data, axis=0))
    print("  mean  :", np.mean(data, axis=0))

    print("\n--- Along axis=1 (row-wise) ---")
    print("  sum   :", np.sum(data, axis=1))
    print("  mean  :", np.mean(data, axis=1))

    # Useful functions
    arr = np.array([3, 1, 4, 1, 5, 9, 2])
    print("\n--- Index-based ---")
    print("  argmin:", np.argmin(arr), "(value:", arr[np.argmin(arr)], ")")
    print("  argmax:", np.argmax(arr), "(value:", arr[np.argmax(arr)], ")")

    # Cumulative
    print("  cumsum :", np.cumsum(arr))
    print("  cumprod:", np.cumprod(np.array([1, 2, 3, 4])))

    # Percentiles
    ages = np.array([22, 25, 28, 30, 35, 40, 45, 50, 60, 70])
    print("\n  25th percentile:", np.percentile(ages, 25))
    print("  75th percentile:", np.percentile(ages, 75))

    # Correlation coefficient
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    print("  Correlation matrix:\n", np.corrcoef(x, y), "\n")


def part_h_linalg():
    """Part H: Linear Algebra."""
    print("=" * 60)
    print("PART H: Linear Algebra")
    print("=" * 60)

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    # Dot product / Matrix multiplication
    print("A @ B (matrix multiply):\n", A @ B)
    print("np.dot(A, B):\n", np.dot(A, B))

    # Transpose
    print("A.T:\n", A.T)

    # Determinant
    print("det(A):", np.linalg.det(A))

    # Inverse
    print("inv(A):\n", np.linalg.inv(A))

    # Eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("Eigenvalues:", eigenvalues)

    # Solving linear systems: Ax = b
    b = np.array([5, 11])
    x = np.linalg.solve(A, b)
    print("Solve Ax = b → x:", x, "\n")


def part_i_random():
    """Part I: Random Number Generation."""
    print("=" * 60)
    print("PART I: Random Number Generation")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)  # Modern recommended way

    print("Random floats [0, 1):", rng.random(5))
    print("Random ints [1, 10) :", rng.integers(1, 10, size=5))
    print("Normal dist (μ=0, σ=1):", rng.normal(0, 1, size=5))
    print("Uniform dist [5, 10) :", rng.uniform(5, 10, size=5))

    # Shuffle
    arr = np.array([1, 2, 3, 4, 5])
    rng.shuffle(arr)
    print("Shuffled:", arr)

    # Choice (sampling)
    choices = rng.choice([10, 20, 30, 40, 50], size=3, replace=False)
    print("Random choice (no replacement):", choices, "\n")


def part_j_sorting():
    """Part J: Sorting & Searching."""
    print("=" * 60)
    print("PART J: Sorting & Searching")
    print("=" * 60)

    arr = np.array([42, 15, 8, 23, 4, 16])
    print("Original:", arr)
    print("Sorted  :", np.sort(arr))
    print("Argsort :", np.argsort(arr), "(indices that would sort the array)")

    # Unique values
    data = np.array([3, 1, 2, 3, 1, 2, 2, 4])
    unique, counts = np.unique(data, return_counts=True)
    print("\nUnique values:", unique)
    print("Counts       :", counts)

    # Searching
    arr = np.array([10, 25, 30, 45, 50])
    idx = np.searchsorted(arr, 35)  # Where would 35 go to keep sorted?
    print("searchsorted(35):", idx, "(insert before index", idx, ")")

    # Clipping values to a range
    data = np.array([-5, 2, 8, 15, 25])
    clipped = np.clip(data, 0, 10)
    print("clip(0, 10):", clipped, "\n")


def part_k_practical():
    """Part K: Practical Examples."""
    print("=" * 60)
    print("PART K: Practical Examples")
    print("=" * 60)

    # 1. Normalize data to [0, 1]
    data = np.array([10, 20, 30, 40, 50], dtype=float)
    normalized = (data - data.min()) / (data.max() - data.min())
    print("1. Min-Max Normalization:", normalized)

    # 2. Standardize data (z-score)
    standardized = (data - data.mean()) / data.std()
    print("2. Z-score Standardization:", standardized)

    # 3. Calculate Euclidean distance between two points
    p1 = np.array([1, 2, 3])
    p2 = np.array([4, 5, 6])
    distance = np.sqrt(np.sum((p1 - p2) ** 2))
    print(f"3. Euclidean distance: {distance:.4f}")

    # 4. One-hot encoding
    labels = np.array([0, 1, 2, 1, 0])
    one_hot = np.eye(3)[labels]
    print("4. One-hot encoding:\n", one_hot)

    # 5. Moving average
    prices = np.array([100, 102, 105, 103, 107, 110, 108, 112])
    window = 3
    moving_avg = np.convolve(prices, np.ones(window)/window, mode='valid')
    print(f"5. Moving average (window={window}):", np.round(moving_avg, 2))

    print()


def main():
    part_a_creation()
    part_b_attributes()
    part_c_reshape()
    part_d_indexing()
    part_e_math()
    part_f_broadcasting()
    part_g_aggregations()
    part_h_linalg()
    part_i_random()
    part_j_sorting()
    part_k_practical()

    print("=" * 60)
    print("🎉 NumPy Deep Dive Complete!")
    print("You now know the full NumPy toolkit for data science.")
    print("=" * 60)

if __name__ == "__main__":
    main()
