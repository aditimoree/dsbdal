import pandas as pd

# Load Dataset
df = pd.read_csv("/mnt/data/Salary_Data.csv")

# View columns
print(df.head())

# Create a new Categorical Column based on 'YearsExperience'
def categorize_experience(exp):
    if exp <= 3:
        return 'Fresher'
    elif exp <= 7:
        return 'Junior'
    else:
        return 'Senior'

df['Experience_Group'] = df['YearsExperience'].apply(categorize_experience)

# Display to verify
print(df[['YearsExperience', 'Experience_Group']].head())

# Group by the new categorical variable and calculate summary statistics
grouped_stats = df.groupby('Experience_Group')['Salary']

# Mean Salary by Experience Group
print("Mean Salary:")
print(grouped_stats.mean())

# Median Salary
print("\nMedian Salary:")
print(grouped_stats.median())

# Minimum Salary
print("\nMinimum Salary:")
print(grouped_stats.min())

# Maximum Salary
print("\nMaximum Salary:")
print(grouped_stats.max())

# Standard Deviation
print("\nStandard Deviation of Salary:")
print(grouped_stats.std())

# All Summary Stats Together
print("\nSummary Statistics:")
print(grouped_stats.agg(['mean', 'median', 'min', 'max', 'std']))

# Create a list: Numeric value (average salary) for each Experience Group
experience_salary_list = grouped_stats.mean().tolist()
print("\nList of Average Salaries by Experience Group:", experience_salary_list)



# Load the Iris Dataset
df1 = pd.read_csv("iris.csv")

# Check for missing values
df1.isnull().sum()

# View first 5 records
df1.head()

# Group by variety and calculate sepal.length statistics
grouped_stats1 = df1.groupby('variety')['sepal.length']

# Aggregate Mean, Median, Min, Max, and Std
print(grouped_stats1.agg(['mean', 'median', 'min', 'max', 'std']))

# Filter data based on Species
Setosa = df1[df1['variety'] == 'Iris-Setosa']
Versicolor = df1[df1['variety'] == 'Iris-Versicolor']
Virginica = df1[df1['variety'] == 'Iris-Virginica']

# Display first few rows (optional)
print(Setosa.head())
print(Versicolor.head())
print(Virginica.head())
