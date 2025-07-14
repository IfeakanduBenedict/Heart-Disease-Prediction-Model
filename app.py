import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the heart disease dataset from the CSV file into a pandas DataFrame
df = pd.read_csv('heart.csv')

# Display the first five rows of the DataFrame to preview the data
df.head()

# Display the last five rows of the DataFrame to preview the data
df.tail()

# Display the shape (number of rows, number of columns) of the DataFrame
df.shape

# Retrieve a list containing the names of all columns in the DataFrame
df.columns.tolist()

# Display summary information about the DataFrame, including column names, data types, and non-null counts
df.info()

# Check for missing values by counting the number of null (NaN) entries in each column
df.isnull().sum()

# Generate summary statistics for all numerical columns in the DataFrame
df.describe()

# Count the occurrences of each class (0 or 1) in the 'target' column and create a bar plot
df['target'].value_counts().plot(kind='bar')

# Set the title of the plot for better understanding
plt.title('Distribution of Target Variable')

# Label the x-axis to indicate what the classes represent (1 = heart disease, 0 = no heart disease)
plt.xlabel('Heart Disease (1=Yes, 0=No)')

# Label the y-axis to show the number of cases in each class
plt.ylabel('Count')

# Display the plot
plt.show()

# Create a histogram of the 'age' column with 20 bins to show how patient ages are distributed
sns.histplot(df['age'], bins=20, kde=True)

# Add a title to the plot to describe what is being visualized
plt.title('Age Distribution')

# Display the histogram plot
plt.show()

# Create a bar plot showing the count of each chest pain type (cp) in the dataset
sns.countplot(x='cp', data=df)

# Add a title to describe the chart
plt.title('Chest Pain Type Distribution')

# Label the x-axis to indicate the chest pain type categories (0 to 3)
plt.xlabel('Chest Pain Type (0-3)')

# Display the plot
plt.show()

# Create a matrix of scatter plots and histograms to visualize pairwise relationships
# between selected numerical features, colored by the 'target' variable
sns.pairplot(df, vars=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], hue='target')

# Display the pairplot
plt.show()

# Define a list of commonly analyzed numeric features to check for outliers
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Set the overall figure size for the group of subplots (width=12, height=6 inches)
plt.figure(figsize=(12, 6))

# Loop through each numeric feature and create a boxplot in a subplot
for i, col in enumerate(numeric_features, 1):
    # Create a subplot in a 2-row, 3-column grid at position i
    plt.subplot(2, 3, i)

    # Create a boxplot for the current column to visualize its distribution and detect outliers
    sns.boxplot(y=df[col])

    # Set the title of the subplot to indicate which feature is being plotted
    plt.title(f'Boxplot of {col}')

# Adjust subplot layout to prevent overlapping of titles and axes
plt.tight_layout()

# Display all the boxplots
plt.show()

# Convert categorical columns ('cp', 'thal', 'slope') into dummy/indicator variables
# drop_first=True avoids multicollinearity by dropping the first category of each feature
df_encoded = pd.get_dummies(df, columns=['cp', 'thal', 'slope'], drop_first=True)

# Display the first five rows of the updated DataFrame with the new dummy variables
df_encoded.head()

# Create the feature matrix (X) by dropping the 'target' column from the encoded DataFrame
# These are the input variables used to train the model
X = df_encoded.drop('target', axis=1)

# Create the target vector (y) by selecting the 'target' column
# This is the output variable the model is trying to predict (1 = heart disease, 0 = no heart disease)
y = df_encoded['target']

# Split the dataset into training and testing sets
# 80% of the data is used for training, and 20% is used for testing
# random_state=42 ensures reproducibility of the split
# stratify=y ensures that the proportion of classes in 'y' (target) is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Print the shape of the training feature set (rows, columns)
print(X_train.shape)

# Print the shape of the testing feature set (rows, columns)
print(X_test.shape)

# Initialize the Logistic Regression model with a maximum of 5000 iterations
# random_state=42 ensures reproducibility of the results
logreg = LogisticRegression(max_iter=5000, random_state=42)

# Fit the Logistic Regression model to the training data (learn the relationship between features and target)
logreg.fit(X_train, y_train)

# Predict outcomes on test data
y_pred = logreg.predict(X_test)

# Calculate the accuracy of the model by comparing predicted and actual labels
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy as a percentage with two decimal places
print(f'Accuracy: {accuracy:.2f}')

# Generate a confusion matrix to evaluate model performance across all classification outcomes
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap for easier interpretation
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

# Label the x-axis with 'Predicted' class values
plt.xlabel('Predicted')

# Label the y-axis with 'Actual' class values
plt.ylabel('Actual')

# Add a title to the heatmap
plt.title('Confusion Matrix')

# Display the plot
plt.show()

# Print detailed classification metrics: precision, recall, f1-score, and support for each class
print(classification_report(y_test, y_pred))

