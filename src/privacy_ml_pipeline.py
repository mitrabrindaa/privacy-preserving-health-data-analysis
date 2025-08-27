import torch as th
import syft as sy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the differentially private data
df = pd.read_csv('../data/DP_Disease_symptom_and_patient_profile.csv')


# Encode categorical features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# One-hot encode categorical features
X = pd.get_dummies(X)


# Analyze target variable distribution
print('Target variable class counts:')
print(df['Disease'].value_counts())


# Group rare diseases into 'Other' category (increase threshold)
rare_threshold = 10
disease_counts = df['Disease'].value_counts()
rare_diseases = disease_counts[disease_counts < rare_threshold].index
df['Disease_grouped'] = df['Disease'].apply(lambda x: x if x not in rare_diseases else 'Other')


# Automatically use all available features except the target for modeling
feature_cols = [col for col in df.columns if col not in ['Disease', 'Disease_grouped']]
X = df[feature_cols]
y = df['Disease_grouped']

# Encode categorical features
X = pd.get_dummies(X)

# Encode target labels as integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the dataset using RandomOverSampler
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Reset indices after balancing
X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
y_train_sr = pd.Series(y_train).reset_index(drop=True)

# Convert boolean columns to integers
for col in X_train_df.select_dtypes(include=['bool']).columns:
	X_train_df[col] = X_train_df[col].astype(int)

# Convert data to tensors
X_train_tensor = th.tensor(X_train_df.values).float()
y_train_tensor = th.tensor(y_train_sr.values).long()

# Encode target labels as integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['Disease_grouped'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Reset indices after train-test split
X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
y_train_sr = pd.Series(y_train).reset_index(drop=True)
# Create mask and apply to both
mask = X_train_df.notnull().all(axis=1)
X_train_df = X_train_df[mask].reset_index(drop=True)
y_train_sr = y_train_sr[mask].reset_index(drop=True)

# Print column types before conversion
print('Column types before conversion:')
print(X_train_df.dtypes)

# Convert all columns to numeric, drop non-numeric columns
X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce')
non_numeric_cols = X_train_df.columns[X_train_df.dtypes == 'object']
if len(non_numeric_cols) > 0:
	print(f'Dropping non-numeric columns: {list(non_numeric_cols)}')
	X_train_df = X_train_df.drop(columns=non_numeric_cols)

# Drop any remaining NaNs
X_train_df = X_train_df.dropna().reset_index(drop=True)
y_train_sr = y_train_sr.loc[X_train_df.index].reset_index(drop=True)

# Print column types after conversion
print('Column types after conversion:')
print(X_train_df.dtypes)

# Convert boolean columns to integers
for col in X_train_df.select_dtypes(include=['bool']).columns:
	X_train_df[col] = X_train_df[col].astype(int)

# Convert data to tensors
X_train_tensor = th.tensor(X_train_df.values).float()
y_train_tensor = th.tensor(y_train_sr.values).long()

# SMPC/federated learning setup is deprecated in latest Syft.
# For privacy-preserving computation, refer to Syft documentation:
# https://github.com/OpenMined/PySyft
# The following code uses standard PyTorch for local tensor operations.
# Privacy-preserving sharing logic should be implemented here using modern Syft or PyTorch features.
# See https://github.com/OpenMined/PySyft for up-to-date examples and documentation.

# Train a RandomForest model with hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {
	'n_estimators': [100, 200],
	'max_depth': [5, 10, None],
	'min_samples_split': [2, 5],
	'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_tensor.numpy(), y_train_tensor.numpy())
print(f'Best parameters: {grid_search.best_params_}')

# Preprocess X_test
X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
for col in X_test_df.select_dtypes(include=['bool']).columns:
	X_test_df[col] = X_test_df[col].astype(int)

# Predict and evaluate on the test set
y_pred = grid_search.predict(X_test_df.values)
print("Accuracy:", accuracy_score(y_test, y_pred))
