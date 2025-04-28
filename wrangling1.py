import pandas as pd 
import numpy as np
import sklearn.preprocessing 
import LabelEncoder

df = pd.read_csv('Iris.csv')
df.dtypes
df.head()
df.info()
df.describe()
df.isnull().sum()
df.shape

# Example of type conversion
df['SepalLengthCm'] = df['SepalLengthCm'].astype(float)


le =LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])
df[['Species', 'Species_encoded']].head()
