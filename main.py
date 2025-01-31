import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:\\Users\\Waithera\\Downloads\\File.csv")

# Check the first few rows
print("First 5 rows of the dataset:\n", data.head())

# Check column names
print("Column names before cleaning:", data.columns)

# Strip spaces and handle potential special characters
data.columns = data.columns.str.strip()

# If 'Date/Time' exists, rename it to 'Date'
if 'Date/Time' in data.columns:
    data.rename(columns={'Date/Time': 'Date'}, inplace=True)

# Confirm column names after renaming
print("Column names after renaming:", data.columns)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Check for missing values
print("Missing values count:\n", data.isnull().sum())

# Display dataset info
print("Dataset Information:")
print(data.info())

# Display summary statistics
print("Summary Statistics:\n", data.describe())

# Drop non-numeric columns ('Date' and 'Weather')
data.drop(columns=['Date', 'Weather'], inplace=True, errors='ignore')

# Handle missing values by filling with column mean
data.fillna(data.mean(), inplace=True)

# Compute and display correlation matrix
corr_matrix = data.corr()
print("Correlation Matrix:\n", corr_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Select features (independent variables) and target (dependent variable)
X = data.drop(columns=['Temp_C'])  # Features: all except 'Temp_C'
y = data['Temp_C']  # Target variable: Temperature

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs. predicted temperature
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Temperature")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle="--")  # Ideal line
plt.show()

# Plot the temperature trend over time
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Temp_C'], label="Temperature (°C)", color='blue')
plt.xlabel("Time")
plt.ylabel("Temperature (Celsius)")
plt.title("Temperature Trend Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Save cleaned dataset (if necessary)
data.to_csv("cleaned_File.csv", index=False)