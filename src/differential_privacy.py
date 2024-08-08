import pandas as pd
import numpy as np

# Load the preprocessed data
df = pd.read_csv('data/Preprocessed_Disease_symptom_and_patient_profile.csv')

# Example: Applying differential privacy by adding Laplace noise
def add_laplace_noise(data, epsilon=0.1):
    scale = 1 / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

# Apply noise to numerical columns
numerical_columns = df.select_dtypes(include=np.number).columns
df[numerical_columns] = add_laplace_noise(df[numerical_columns])

# Save the differentially private data
df.to_csv('data/DP_Disease_symptom_and_patient_profile.csv', index=False)
print("Differential privacy applied and data saved.")
