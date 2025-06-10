import pandas as pd

# Load the parquet file
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')

# The dataset typically has tpep_pickup_datetime and tpep_dropoff_datetime columns
# Calculate duration in minutes
if 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
    # Calculate duration in minutes
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Remove outliers (negative durations or extremely long trips over 24 hours)
    df = df[(df['duration'] >= 0) & (df['duration'] <= 24 * 60)]

    # Calculate standard deviation
    duration_std = df['duration'].std()

    print(f"Standard deviation of trip duration in January 2023: {duration_std:.2f} minutes")
    print(f"Mean trip duration: {df['duration'].mean():.2f} minutes")
    print(f"Median trip duration: {df['duration'].median():.2f} minutes")
    print(f"Number of trips analyzed: {len(df)}")
else:
    # Check what columns are available if expected columns aren't found
    print("Expected time columns not found. Available columns:")
    print(df.columns.tolist())