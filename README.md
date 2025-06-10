# NYC Taxi Trip Duration Prediction

This repository contains solutions to a MLOps Zoomcamp homework assignment that predicts taxi trip durations using the NYC Yellow Taxi dataset. The project demonstrates a complete machine learning workflow from data loading and cleaning to model training and evaluation.

## Solutions to Homework Questions

### Q1. Downloading the data

**Question:** How many columns are there in the Yellow Taxi Trip Records for January 2023?

**Solution:** 
```python
import pandas as pd

# Load the Parquet file
df = pd.read_parquet('yellow_tripdata_2023-01.parquet', engine='pyarrow')

# Get the number of columns
num_columns = len(df.columns)
print(f"The Parquet file contains {num_columns} columns")
```

**Answer:** 19 columns

The Yellow Taxi dataset for January 2023 was loaded and the number of columns was counted. This provides a first look at the dataset's structure.

### Q2. Computing duration

**Question:** What's the standard deviation of the trips duration in January?

**Solution:**
```python
import pandas as pd

# Calculate duration in minutes
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

# Remove outliers (negative durations or extremely long trips over 24 hours)
df = df[(df['duration'] >= 0) & (df['duration'] <= 24 * 60)]

# Calculate standard deviation
duration_std = df['duration'].std()
print(f"Standard deviation of trip duration in January 2023: {duration_std:.2f} minutes")
```

**Answer:** 41.54 minutes, which is closest to **42.59** among the given options

A new feature 'duration' was created by calculating the time difference between pickup and dropoff timestamps, then converted to minutes. After removing extreme outliers (negative durations and trips longer than 24 hours), the standard deviation was calculated to understand the variability in trip times.

### Q3. Dropping outliers

**Question:** What fraction of the records left after dropping trips that aren't between 1 and 60 minutes?

**Solution:**
```python
# Initial filtering to remove negative durations or extremely long trips
df_first_filter = df[(df['duration'] >= 0) & (df['duration'] <= 24*60)]
count_first_filter = len(df_first_filter)

# Second filter: Keep only trips between 1 and 60 minutes (inclusive)
df_second_filter = df_first_filter[(df_first_filter['duration'] >= 1) & (df_first_filter['duration'] <= 60)]
count_second_filter = len(df_second_filter)

# Calculate the fraction relative to the first filter
fraction = count_second_filter / count_first_filter
print(f"Fraction of records left: {fraction:.4f} ({fraction*100:.2f}%)")
```

**Answer:** 98.12%, which is closest to **98%**

This step is crucial for data cleaning. Trips that were unreasonably short (under 1 minute) or unusually long (over 60 minutes) were removed. The high percentage of remaining records (98.12%) indicates that the filtering criteria captured most typical taxi trips.

### Q4. One-hot encoding

**Question:** What's the dimensionality of the feature matrix after one-hot encoding pickup and dropoff location IDs?

**Solution:**
```python
from sklearn.feature_extraction import DictVectorizer

# Select only location IDs
df_locations = df[['PULocationID', 'DOLocationID']]

# Convert to list of dictionaries with string IDs
dict_list = df_locations.astype(str).to_dict(orient='records')

# Create and fit the DictVectorizer
dv = DictVectorizer(sparse=True)
X = dv.fit_transform(dict_list)

# Get the dimensionality
n_features = X.shape[1]
print(f"Dimensionality of the feature matrix: {n_features}")
```

**Answer:** **515**

One-hot encoding transformed the categorical location IDs into a format machine learning algorithms can process. Each unique pickup and dropoff location ID got its own column, resulting in 515 features. This dimensionality indicates there are approximately 257-258 unique taxi zones in NYC.

### Q5. Training a model

**Question:** What's the RMSE of a linear regression model trained on the January data?

**Solution:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Extract the target variable
y = df['duration'].values

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on training data
y_pred = model.predict(X)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE on training data: {rmse:.4f}")
```

**Answer:** 7.6493, which is closest to **7.64**

A linear regression model was trained to predict trip durations based only on pickup and dropoff locations. The RMSE of 7.65 minutes indicates that, on average, the model's predictions are about 7.65 minutes off from the actual trip durations in the training data.

### Q6. Evaluating the model

**Question:** What's the RMSE on the February 2023 validation data?

**Solution:**
```python
# Load and process February data
df_val = pd.read_parquet('yellow_tripdata_2023-02.parquet')
df_val['duration'] = (df_val['tpep_dropoff_datetime'] - df_val['tpep_pickup_datetime']).dt.total_seconds() / 60
df_val = df_val[(df_val['duration'] >= 1) & (df_val['duration'] <= 60)]
y_val = df_val['duration'].values

# Prepare features using the same vectorizer
df_val_locations = df_val[['PULocationID', 'DOLocationID']]
dict_val_list = df_val_locations.astype(str).to_dict(orient='records')
X_val = dv.transform(dict_val_list)

# Make predictions
y_val_pred = model.predict(X_val)

# Calculate validation RMSE
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE on validation data: {rmse_val:.4f}")
```

**Answer:** 7.8118, which is closest to **7.81**

The true test of a model is how well it performs on new data. The model was applied to February taxi data and an RMSE of 7.81 minutes was calculated. The small difference between training RMSE (7.65) and validation RMSE (7.81) indicates the model generalizes well to new data.

## Project Structure

- `q1.py`: Loads the dataset and counts columns
- `q2.py`: Calculates duration standard deviation
- `q3.py`: Filters outliers and calculates remaining fraction
- `q4.py`: Performs one-hot encoding and reports dimensionality
- `q5.py`: Trains linear regression model and calculates training RMSE
- `q6.py`: Applies model to validation data and calculates validation RMSE

## Insights and Observations

This project demonstrates a complete machine learning workflow:

1. **Data Preparation**: Trip durations were calculated and outliers were removed to focus on typical taxi trips.

2. **Feature Engineering**: Categorical location IDs were transformed into a machine-readable format using one-hot encoding.

3. **Model Training**: A linear regression model was built that learns how pickup and dropoff locations relate to trip durations.

4. **Validation**: The model was tested on February data to ensure it generalizes well to new trips.

Despite using only location information (no time of day, weather, or traffic data), the model achieved reasonable accuracy with an RMSE of about 7.8 minutes. The small difference between training and validation RMSE (only about 0.16 minutes) suggests the model captures stable patterns in how locations influence trip durations.

## Dependencies

The project requires:
- pandas>=2.2.3
- pyarrow>=20.0.0
- scikit-learn>=1.6.1
- parquet-tools>=0.2.16

These dependencies are listed in the `pyproject.toml` file.