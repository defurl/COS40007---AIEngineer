# Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Understanding the Data ---

try:
    data = pd.read_csv('concrete.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'concrete.csv' file not found.")

print("\n----- First rows of Dataset -----")
# Display the first few rows of the dataset
print(data.head())

# Display summary statistics of the dataset
print("\n----- Summary Statistics -----")
print(data.info())

print(f"\nDataset Shape:", data.shape)

data = data.drop_duplicates()
print(f"Dataset Shape after removing duplicates:", data.shape)

print("Generating boxplots to visualize outliers...")
# Generate boxplots for each feature to visualize outliers
data.plot(kind='box', subplots=True, layout=(3,4), figsize=(15,10))
plt.tight_layout()
plt.show()