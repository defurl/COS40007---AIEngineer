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

# --- Studio Activity 2: Data Cleaning ---
print("--- Starting Data Cleaning ---")

# Check for and remove duplicate rows
duplicates_count = df.duplicated().sum()
print(f"\nFound {duplicates_count} duplicate rows.")
if duplicates_count > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate rows have been removed.")
    print(f"Dataset shape after removing duplicates: {df.shape[0]} rows, {df.shape[1]} columns")

# Visualize outliers before removal
print("\nGenerating boxplots to visualize outliers before removal...")
plt.figure(figsize=(18, 10))
plt.boxplot(df.values, labels=df.columns)
plt.title('Boxplots of Concrete Components (Before Outlier Removal)')
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Remove outliers using the Interquartile Range (IQR) method
print("\nRemoving outliers based on the IQR method...")
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

print(f"Dataset shape after removing outliers: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
print(f"Number of rows removed: {df.shape[0] - df_cleaned.shape[0]}")

# Visualize the data again after outlier removal
print("\nGenerating boxplots to visualize data after outlier removal...")
plt.figure(figsize=(18, 10))
plt.boxplot(df_cleaned.values, labels=df_cleaned.columns)
plt.title('Boxplots of Concrete Components (After Outlier Removal)')
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
print("--- Data Cleaning Complete ---\n")


# --- Studio Activity 3: Exploratory Data Analysis (EDA) ---
print("--- Starting Exploratory Data Analysis ---")

# Identify Target and Predictors
target_variable = 'strength'
predictors = df_cleaned.columns.drop(target_variable)

print(f"\nTarget Variable: {target_variable}")
print(f"Predictor Variables: {list(predictors)}")

# Univariate Analysis: Generate histograms for each variable
print("\nGenerating histograms for Univariate Analysis...")
df_cleaned.hist(figsize=(20, 15), bins=30, edgecolor='black')
plt.suptitle('Distribution of Each Variable in the Concrete Dataset', size=20)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Summary Statistics
print("\n--- Summary Statistics of the Cleaned Dataset ---")
summary_stats = df_cleaned.describe()
print(summary_stats)

# Multivariate Analysis: Correlation Matrix and Heatmap
print("\n--- Performing Multivariate Analysis ---")
correlation_matrix = df_cleaned.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
cax = plt.matshow(correlation_matrix, fignum=plt.gcf().number, cmap='viridis')
plt.colorbar(cax)

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

# Add correlation values to the heatmap
for i, col_i in enumerate(correlation_matrix.columns):
    for j, col_j in enumerate(correlation_matrix.columns):
        value = correlation_matrix.loc[col_i, col_j]
        color = 'white' if abs(value) > 0.5 else 'black'
        plt.text(j, i, f'{value:.2f}', ha='center', va='center', color=color)

plt.title('Correlation Matrix of Concrete Components', pad=80)
plt.tight_layout()
plt.show()
print("\n--- Exploratory Data Analysis Complete ---")
