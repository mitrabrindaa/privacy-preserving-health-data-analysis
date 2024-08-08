import torch as th
import syft as sy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the differentially private data
df = pd.read_csv('data/DP_Disease_symptom_and_patient_profile.csv')

# Split data into features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the data types of your DataFrame
print(df.dtypes)

# Convert all columns to numerical types if possible
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values if they exist
df.dropna(inplace=True)

# Convert to PyTorch tensor
X_train = th.tensor(df.values).float()


# Convert data to tensors
X_train = th.tensor(X_train.values).float()
y_train = th.tensor(y_train.values).long()

# Setup SMPC
hook = sy.TorchHook(th)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# Share the data between the workers
X_train_shared = X_train.share(alice, bob)
y_train_shared = y_train.share(alice, bob)

# Train a simple model (e.g., RandomForest) on shared data
# (This is a simplified example; SMPC usually involves more complex procedures)
model = RandomForestClassifier()
model.fit(X_train_shared.get().numpy(), y_train_shared.get().numpy())

# Predict and evaluate on the test set
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
