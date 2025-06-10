import pandas as pd

# Load the parquet file
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')

# Calculate duration in minutes
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

# Initial count before any filtering
total_count = len(df)

# First filter: Remove negative durations or extremely long trips (as done previously)
df_first_filter = df[(df['duration'] >= 0) & (df['duration'] <= 24*60)]
count_first_filter = len(df_first_filter)

# Second filter: Keep only trips between 1 and 60 minutes (inclusive)
df_second_filter = df_first_filter[(df_first_filter['duration'] >= 1) & (df_first_filter['duration'] <= 60)]
count_second_filter = len(df_second_filter)

# Calculate the fraction relative to the first filter
fraction = count_second_filter / count_first_filter

print(f"Original count (after first filter): {count_first_filter}")
print(f"Count after second filter (1-60 minutes): {count_second_filter}")
print(f"Fraction of records left: {fraction:.4f} ({fraction*100:.2f}%)")