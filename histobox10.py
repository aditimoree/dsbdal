# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("iris.csv")

# Display first 5 rows
data.head()

# Display dataset information
data.info()

# Display summary statistics
data.describe()

# Check for missing values
data.isnull().sum()

# Check shape of dataset
data.shape

# Display datatypes of each column
data.dtypes

# Display last 5 rows
data.tail()

# Plot histograms for each numerical column
data.hist()
plt.show()

# Boxplot for all columns
data.boxplot()
plt.show()

# Seaborn boxplot with aesthetic styling
sb.boxplot(data=data, palette="Set2", saturation=0.75)

# Scatter plot between Sepal Length and Sepal Width
plt.scatter(data["sepal_length"], data["sepal_width"])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Boxplot of Sepal Length grouped by Species
sb.boxplot(data=data, x="sepal_length", y="species")
plt.title('Distribution of sepal length')

# Outlier detection using IQR
Q1 = data.drop(columns=['species']).quantile(0.25)
Q3 = data.drop(columns=['species']).quantile(0.75)
IQR = Q3 - Q1

outliers = ((data.drop(columns=['species']) < (Q1 - 1.5 * IQR)) | 
            (data.drop(columns=['species']) > (Q3 + 1.5 * IQR)))

# Count of outliers in each column
outlier_counts = outliers.sum()
print(outlier_counts)

# Remove outliers
df_cleaned = data[~outliers.any(axis=1)]

# Show dataset shape before and after outlier removal
print("Dataset before removing outliers:", data.shape)
print("Dataset after removing outliers:", df_cleaned.shape)
