import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic.csv")
df.head()
df.isnull().sum()
df.dtypes
print(df.columns)
df.columns = df.columns.str.strip()

sns.catplot(x='Sex', hue='Survived', kind='count', data=df)

pclass_survived = df.groupby(['Pclass', 'Survived']).size().unstack()
sns.heatmap(pclass_survived, annot=True, fmt='d')

sns.violinplot(x='Sex', y='Age', hue='Survived', data=df, split=True)

df['Age'].fillna(df['Age'].mean(), inplace=True)

sns.histplot(df['Fare'])

sns.distplot(df['Fare'])  # deprecated
sns.distplot(df['Fare'], kde=False)

sns.jointplot(x=df['Age'], y=df['Fare'], kind='scatter')
sns.jointplot(x=df['Age'], y=df['Fare'], kind='hex')

sns.rugplot(df['Fare'])

sns.barplot(x='Sex', y='Age', data=df, estimator=np.std)

sns.countplot(x='Sex', data=df)

sns.boxplot(x='Sex', y='Age', data=df)
sns.boxplot(x='Sex', y='Age', data=df, hue='Survived')

sns.stripplot(x='Sex', y='Age', data=df, jitter=True, hue='Survived')

