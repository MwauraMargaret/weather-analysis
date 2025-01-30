import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:\\Users\\Waithera\\Downloads\\File.csv")

# Strip spaces from column names to avoid key errors
data.columns = data.columns.str.strip()

# Rename 'Date/Time' to 'Date' if present
if 'Date/Time' in data.columns:
    data.rename(columns={'Date/Time': 'Date'}, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])

# Print first few rows to verify data
print("First few rows of data:\n", data.head())

# Print column names
print("Column names:\n", data.columns)

# Check for missing values
print("Missing values per column:\n", data.isnull().sum())

# Select only numeric columns (ignore 'Weather' or other non-numeric columns)
numeric_data = data.select_dtypes(include=['number'])

# Compute correlation matrix
corr_matrix = numeric_data.corr()

# Print correlation matrix
print("Correlation Matrix:\n", corr_matrix)

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Weather Data Correlation Heatmap")
plt.show()
