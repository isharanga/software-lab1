# software-lab1
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

from sklearn.datasets import load_iris
# Load the Iris dataset
data = load_iris()
# Convert to a pandas DataFrame
iris = pd.DataFrame(data=data.data, columns=data.feature_names)
iris['target'] = data.target
# Show the first few rows of the dataset
iris.head()

# Data Summary
iris.describe()
# Data Visualization - Pairplot to show relationships between features
sns.pairplot(iris, hue='target')
plt.show()

# Split the data into features (X) and target (y)
X = iris.drop('target', axis=1) # Features (all columns except the target)
y = iris['target'] # Target (the species)
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Display the shape of the train and test sets
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Create a Logistic Regression model
model = LogisticRegression(max_iter=200)
# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model using accuracy score and classification report
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Sample new data (sepal length, sepal width, petal length, petal width)
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
# Predict the class for the new data
prediction = model.predict(new_data)
predicted_class = data.target_names[prediction][0]
print("Predicted Class: {predicted_class}")
