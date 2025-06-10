import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# First, we need to have our trained model and vectorizer from January
# (I'm including the January training code again for completeness)

# Load the January data and prepare it
df_train = pd.read_parquet('yellow_tripdata_2023-01.parquet')
df_train['duration'] = (df_train['tpep_dropoff_datetime'] - df_train['tpep_pickup_datetime']).dt.total_seconds() / 60
df_train = df_train[(df_train['duration'] >= 1) & (df_train['duration'] <= 60)]
y_train = df_train['duration'].values

# Prepare features for training data
df_train_locations = df_train[['PULocationID', 'DOLocationID']]
dict_train_list = df_train_locations.astype(str).to_dict(orient='records')
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(dict_train_list)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Now, load and process the February validation data
df_val = pd.read_parquet('yellow_tripdata_2023-02.parquet')
df_val['duration'] = (df_val['tpep_dropoff_datetime'] - df_val['tpep_pickup_datetime']).dt.total_seconds() / 60
df_val = df_val[(df_val['duration'] >= 1) & (df_val['duration'] <= 60)]
y_val = df_val['duration'].values

# Prepare features for validation data
df_val_locations = df_val[['PULocationID', 'DOLocationID']]
dict_val_list = df_val_locations.astype(str).to_dict(orient='records')

# Transform validation features using the SAME DictVectorizer
# Notice we use transform(), not fit_transform()
X_val = dv.transform(dict_val_list)

# Make predictions
y_val_pred = model.predict(X_val)

# Calculate validation RMSE
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE on validation data: {rmse_val:.4f}")