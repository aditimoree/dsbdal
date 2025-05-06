# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

print("First 5 rows of the dataset:")
print(df.head())

# Encode target variable
df['species'] = df['species'].astype('category').cat.codes

# Split features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Calculate TP, FP, FN, TN
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (TP + FP + FN)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='micro')  # Micro average for multiclass
recall = recall_score(y_test, y_pred, average='micro')

# Print metrics
print("\nPerformance Metrics:")
print(f"True Positives (TP) per class: {TP}")
print(f"False Positives (FP) per class: {FP}")
print(f"True Negatives (TN) per class: {TN}")
print(f"False Negatives (FN) per class: {FN}\n")

print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# ----- Plot 1: Confusion Matrix -----
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Setosa", "Versicolor", "Virginica"], yticklabels=["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()

# ----- Plot 2: Results table (like df.head()) -----
# Create a DataFrame to show predictions
results = pd.DataFrame(X_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
results['Actual'] = y_test
results['Predicted'] = y_pred

# Display first 5 predictions as a table plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
tbl = pd.plotting.table(ax, results.head(), loc='center', cellLoc='center', colWidths=[0.2]*len(results.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.title('Naïve Bayes Predictions (First 5 Rows)', pad=20)
plt.show()
