import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Fix the import for matplotlib

# Load the data from a CSV file
data = pd.read_csv("data.csv")

# Exploratory Data Analysis
data.head()

# Display the shape, info, and summary statistics of the data
data.shape
data.info()
data.describe()

# Check for missing values
data.isnull().sum()

# Count the occurrences of each class in the target variable 'y'
data['y'].value_counts()

# Visualize the distribution of classes using a countplot
sns.countplot(data['y'])

# Check the data types of each column
print(data.dtypes)

# Identify and exclude non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number])

# Plot the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Drop the 'Unnamed' column
data.drop(['Unnamed'], axis=1, inplace=True)

# Extract features (X) and target variable (y)
d = pd.DataFrame(data.iloc[:, 0:-1])
d.head()

# Standardize the features using StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(d)
X = scaled_data
y = data['y']

# Split the data into train/test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Using RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
model = rf.fit(X_train, y_train)

# Make predictions on the test set
pred = model.predict(X_test)

# Evaluate the model using confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))

# Plot the Confusion Matrix using Seaborn
conf_matrix = confusion_matrix(y_test, pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Display the Classification Report
print("Classification Report:\n", classification_report(y_test, pred))



