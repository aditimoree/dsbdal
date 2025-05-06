import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("titanic.csv")

# Show first 5 rows
data.head()

# Dataset structure
data.info()

# Summary statistics
data.describe()

# Show missing values
data.isnull()

# Count of missing values
data.isnull().sum()

# Fill missing values using backfill method
data = data.bfill()

# Recheck for missing values
data.isnull().sum()

# Boxplot of Age by Sex and Survived status
sns.boxplot(data=data, x="Sex", y="Age", hue="Survived")
plt.xlabel("Sex")
plt.ylabel("Age")

# Calculate mean and standard deviation for Age
mean_age = data['Age'].mean()
print(mean_age)

std_age = data['Age'].std()
print(std_age)

# Calculate z-score for Age column
data['zscore'] = (data['Age'] - mean_age) / std_age
print(data['zscore'])

# Identify outliers using z-score (threshold > 3 or < -3)
outliers = data[np.abs(data['zscore']) > 3]
print(outliers[['Age', 'Sex', 'Survived', 'zscore']])

# Remove outliers
titanic_cleaned = data[np.abs(data['zscore']) <= 3]
titanic_cleaned = titanic_cleaned.drop(columns=['zscore'])

# Display dataset sizes
print("Original dataset size:", data.shape[0])
print("Cleaned dataset size:", titanic_cleaned.shape[0])

# Show first 5 rows of cleaned dataset
titanic_cleaned.head()
