# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GgvqADU5xKuF1kwOQPbk7_Z-_elH49a1
"""

import pandas as pd

# Read the data from the CSV file
df = pd.read_csv("/content/Raw_StudentsPerformance_Data.csv")  
# Check for and handle null values
df = df.dropna()


# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Check data types
print("\nData Types:")
print(df.dtypes)


# Explore the dataset
print("\nDataset Summary:")
print(df.describe())
print("\nDataset Head:")
print(df.head())

# Save the cleaned dataset to a CSV file
df.to_csv("/content/Cleaned_StudentsPerformance.csv", index=False)