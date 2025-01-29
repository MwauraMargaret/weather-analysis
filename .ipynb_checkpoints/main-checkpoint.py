import pandas as pd

data = pd.read_csv("C:\Users\Waithera\Downloads\File.csv")

print(data.head())
print(data.info())
print(data.describe())

data.to_csv("cleaned_File.csv", index=False)