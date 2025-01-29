import pandas as pd 
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("C:\\Users\\Waithera\\Downloads\\File.csv")

#Check the first few rows
print(data.head())

#Check column names
print(data.columns)

# Check for missing values
print(data.isnull().sum())

print(data.info())


# Check the exact column names for any discrepancies
print("Column names:", data.columns)

# Strip spaces and handle potential special characters
data.columns = data.columns.str.strip()

# If 'Date/Time' exists, rename it to 'Date'
if 'Date/Time' in data.columns:
    data.rename(columns={'Date/Time': 'Date'}, inplace=True)

# Confirm column names after renaming
print("After renaming:", data.columns)

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Plot the temperature trend
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Temp_C'], label="Temperature (°C)", color='blue')
plt.xlabel("Date")
plt.ylabel("Temperature (Celsius)")
plt.title("Temperature Trend Over Time")
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()
