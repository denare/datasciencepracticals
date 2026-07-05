# %% [markdown]
# # Lesson 1: NumPy Foundations 🧮
# NumPy (Numerical Python) is the foundational library for scientific computing in Python.
# It provides high-performance multi-dimensional array objects and tools for working with them.
# Almost all data science libraries (Pandas, Scikit-learn, SciPy) are built on top of NumPy.
#
# Let's learn the core concepts:
# 1. Creating Arrays
# 2. Array Attributes (shape, dtype)
# 3. Slicing & Indexing
# 4. Element-wise Math & Broadcasting
# 5. Common Aggregations (mean, sum, std)
# 6. Exercises for you!

# pyrefly: ignore [missing-import]
import numpy as np

def main():
    print("=== NumPy Foundations Lesson ===")
    
    # -------------------------------------------------------------------------
    # 1. Creating Arrays
    # -------------------------------------------------------------------------
    print("\n--- 1. Creating Arrays ---")
    # From a list
    list1 = [1, 2, 3, 4, 5]
    arr1d = np.array(list1)
    print("1D Array from list:", arr1d)
    
    # 2D Array (Matrix)
    list2d = [[1, 2, 3], [4, 5, 6]]
    arr2d = np.array(list2d)
    print("2D Array (Matrix):\n", arr2d)
    
    # Helper functions to create common arrays
    zeros = np.zeros((2, 3))       # Array of zeros with shape (2,3)
    ones = np.ones((3, 2))         # Array of ones with shape (3,2)
    arange = np.arange(0, 10, 2)   # Values from 0 up to (but not including) 10, step 2
    linspace = np.linspace(0, 1, 5) # 5 evenly spaced values between 0 and 1
    
    print("Zeros:\n", zeros)
    print("Arange:", arange)
    print("Linspace:", linspace)

    # -------------------------------------------------------------------------
    # 2. Array Attributes
    # -------------------------------------------------------------------------
    print("\n--- 2. Array Attributes ---")
    print("2D Array Shape:", arr2d.shape)  # Returns (rows, columns)
    print("2D Array Data Type:", arr2d.dtype)  # e.g. int64
    print("2D Array Dimensions:", arr2d.ndim)  # number of dimensions (2)

    # -------------------------------------------------------------------------
    # 3. Indexing & Slicing
    # -------------------------------------------------------------------------
    print("\n--- 3. Indexing & Slicing ---")
    # Syntax: array[row_slice, col_slice]
    print("Original 2D Matrix:\n", arr2d)
    print("Get element at row 0, col 1:", arr2d[0, 1])
    print("Get first row entirely:", arr2d[0, :])
    print("Get last column entirely:", arr2d[:, -1])
    print("Slice first two columns of all rows:\n", arr2d[:, :2])

    # -------------------------------------------------------------------------
    # 4. Element-wise Math & Broadcasting
    # -------------------------------------------------------------------------
    print("\n--- 4. Element-wise Math & Broadcasting ---")
    a = np.array([10, 20, 30])
    b = np.array([1, 2, 3])
    
    print("a + b =", a + b)
    print("a * b =", a * b)
    print("a / b =", a / b)
    print("a ** 2 =", a ** 2)
    
    # Broadcasting: Math operations between arrays of different shapes (or array & scalar)
    print("a + 5 =", a + 5)  # The scalar 5 is "broadcasted" to match the shape of a

    # -------------------------------------------------------------------------
    # 5. Common Aggregations
    # -------------------------------------------------------------------------
    print("\n--- 5. Common Aggregations ---")
    data = np.array([[1, 2], [3, 4]])
    print("Data Matrix:\n", data)
    print("Sum of all elements:", np.sum(data))
    print("Mean of all elements:", np.mean(data))
    print("Sum along columns (axis 0):", np.sum(data, axis=0))
    print("Sum along rows (axis 1):", np.sum(data, axis=1))

    # -------------------------------------------------------------------------
    # YOUR TURN! (Exercises)
    # Run the script first, then write your code in the functions below.
    # -------------------------------------------------------------------------
    print("\n==============================================")
    print("Now run the exercises below by calling them in main()!")
    print("==============================================")
    
    # Call your exercises here once you fill them in:
    run_exercises()

def run_exercises():
    print("\n--- Running Exercises ---")
    
    # Exercise 1: Create a 1D NumPy array containing numbers from 10 to 50 (inclusive).
    # Hint: Use np.arange() or np.array()
    # TODO: Write your code here
    ex1 = None  
    print("Exercise 1 Array:", ex1)
    
    # Exercise 2: Create a 3x3 matrix containing numbers from 1 to 9, and extract the middle column.
    # Hint: Use np.arange().reshape(3, 3) then slice it.
    # TODO: Write your code here
    matrix_3x3 = None
    middle_column = None
    print("Exercise 2 Matrix:\n", matrix_3x3)
    print("Exercise 2 Middle Column:", middle_column)
    
    # Exercise 3: Compute the mean and standard deviation of the array [12, 15, 18, 22, 29, 35].
    # Hint: Use np.mean() and np.std()
    # TODO: Write your code here
    values = None
    mean_val = None
    std_val = None
    print(f"Exercise 3: Mean = {mean_val}, Std Dev = {std_val}")

if __name__ == "__main__":
    main()
