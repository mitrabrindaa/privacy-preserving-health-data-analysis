import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\dell pc\privacy-preserving-health-data-analysis\data\Disease_symptom_and_patient_profile_dataset.csv')

# Display basic info about the dataset
print("Dataset Info:")
print(df.info())
print("\nDataset Head:")
print(df.head())

# Describe the dataset
print("\nDataset Description:")
print(df.describe())

# Handle missing values if any
df.dropna(inplace=True)

# Distribution of Disease
target_column = 'Disease'
if target_column in df.columns:
    # Visualize the distribution of target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df)
    plt.title('Distribution of Target Variable')
    plt.savefig('../result/disease_distribution.png')
    plt.close()
else:
    print(f"Error: '{target_column}' column not found in the dataset.")


# Filter numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Check if there are numeric columns left
if numeric_df.empty:
    print("No numeric columns found for correlation analysis.")
else:
    # Visualize correlations
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('../result/correlation_matrix.png')
    plt.close()


df.to_csv('../data/Preprocessed_Disease_symptom_and_patient_profile.csv', index=False)