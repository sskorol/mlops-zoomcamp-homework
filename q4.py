import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse as sp

# Load the parquet file
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')

# Calculate duration and filter as before
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

# Select only the relevant columns - the location IDs
# For NYC taxi data, these are typically 'PULocationID' and 'DOLocationID'
df_locations = df[['PULocationID', 'DOLocationID']]

# Convert the dataframe to a list of dictionaries
# Important: Convert IDs to strings to ensure they're treated as categories
dict_list = df_locations.astype(str).to_dict(orient='records')

# Sample of what these dictionaries look like
print("Sample dictionary:")
print(dict_list[0])

# Create and fit the DictVectorizer
dv = DictVectorizer(sparse=True)
X = dv.fit_transform(dict_list)

# Get the dimensionality of the feature matrix
n_features = X.shape[1]
print(f"Dimensionality of the feature matrix: {n_features}")

# Additional information about the matrix
print(f"Matrix shape: {X.shape}")
print(f"Matrix type: {type(X)}")
print(f"Matrix is sparse: {sp.issparse(X)}")  # Use scipy.sparse.issparse function
print(f"Matrix density: {X.nnz / (X.shape[0] * X.shape[1]):.6f}")