Privacy-Preserving Health Data Analysis

Overview
This project demonstrates privacy-preserving techniques applied to health data analysis. The goal is to ensure that sensitive health data is analyzed and visualized in a way that maintains privacy using cutting-edge techniques like Differential Privacy, Homomorphic Encryption, and Secure Multiparty Computation (SMPC). This project can be a valuable resource for researchers, data scientists, and developers interested in the intersection of data privacy and health analytics.

Tech Stack
Programming Language: Python
Libraries: NumPy, Pandas, PySyft, PyTorch, Seaborn, Matplotlib
Key Features
Data Anonymization: Implements techniques to anonymize sensitive health data, ensuring individual privacy.

Differential Privacy: Applies differential privacy to the dataset, adding noise to sensitive data to prevent re-identification.

Homomorphic Encryption: Demonstrates how to perform computations on encrypted data, allowing analysis without exposing raw data.

Secure Multiparty Computation (SMPC): Implements SMPC for collaborative analysis between multiple parties, ensuring that no single party has access to the entire dataset.

Privacy-Preserving Visualizations: Provides visualizations that offer insights into the data while preserving the privacy of individuals in the dataset.

Project Structure
'data/'
Contains the health dataset used for analysis.

'notebooks/'
data_exploration.ipynb: Jupyter notebook for initial data exploration and understanding the dataset.
differential_privacy.ipynb: Jupyter notebook demonstrating the application of differential privacy to the dataset.
smpc_analysis.ipynb: Jupyter notebook implementing Secure Multiparty Computation for collaborative analysis.

'src/'
data_exploration.py: Python script for data exploration and initial analysis.
differential_privacy.py: Python script for applying differential privacy to the dataset.
smpc_analysis.py: Python script implementing SMPC techniques