import numpy as npy
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("student.csv")  # <-- Update path if needed
df.head()

# Display first 10 records
df.head(10)

# Dataset Information
df.info()

# Describe the dataset
df.describe()

# Check for missing values
df.isnull()
df.isnull().sum()

# Check for non-missing values
df.notnull()
df.notnull().sum()

# Alternative missing value checking
df.isna()

# Handling Missing Values
df.fillna(1)       # Fill missing values with 1
df.ffill()         # Forward fill
df.bfill()         # Backward fill
df.dropna()        # Drop rows with missing values
df.drop('Age', axis=1)  # Drop 'Age' column
df.replace(to_replace=npy.nan, value='1')  # Replace NaN with 1
df.interpolate()   # Interpolate missing values

# Outlier Detection using Z-Score
z_scores = stats.zscore(df['StudentID'])
print(z_scores)

abs_z_scores = npy.abs(z_scores)
print(abs_z_scores)

# Filter out outliers
outliers = (z_scores > 1.73)
outliers = outliers[:len(df)]  
df = df[~outliers]

# Boxplots
sns.boxplot(df['StudentID'])
sns.boxplot(df['ParentalEducation'])

# Outlier Treatment using IQR
Q1 = df['ParentalEducation'].quantile(0.25)
Q3 = df['ParentalEducation'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['ParentalEducation'] >= lower_bound) & (df['ParentalEducation'] <= upper_bound)]

# Final Boxplot after treatment
sns.boxplot(df['ParentalEducation'])
