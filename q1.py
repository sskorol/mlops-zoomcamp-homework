import pandas as pd

# Load the Parquet file
df = pd.read_parquet('yellow_tripdata_2023-01.parquet', engine='pyarrow')

# Get the number of columns
num_columns = len(df.columns)
print(f"The Parquet file contains {num_columns} columns")

# To see the column names
print("Column names:")
print(df.columns.tolist())