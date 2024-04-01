import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Fix the import for matplotlib

data = pd.read_csv("data.csv")

# Check the data types of each column
print(data.dtypes)

# Identify and exclude non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number])

# Plot the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()
