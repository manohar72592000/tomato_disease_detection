import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
datasets = pd.read_csv('Labled_DATAUpdate1.csv')
print(f"Dataset shape: {datasets.shape}")

# Check for NaN values
print("Checking for missing values...")
missing_values = datasets.isnull().sum()
print(missing_values)

# Handle missing values: Replace NaNs with the column mean
datasets.fillna(datasets.mean(), inplace=True)
print("NaN values replaced with column means.")

# Separate features and labels
X = datasets.iloc[:, 1:].values  # Features
Y = datasets.iloc[:, 0].values   # Labels
print(f"Features shape: {X.shape}, Labels shape: {Y.shape}")

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

# Train SVM model with a linear kernel
clf = svm.SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)
print("SVM model trained successfully!")

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
