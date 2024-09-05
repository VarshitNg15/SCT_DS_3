import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import graphviz
import warnings

warnings.filterwarnings('ignore')

# Load data with the correct delimiter
df = pd.read_csv("bank-full.csv", sep=';')  # Modify sep if necessary

# Check the shape and head of the DataFrame
print(df.shape)
print(df.head())

# Set appropriate column names
df.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
              'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit']

# Proceed with the rest of your operations...


# Check for missing values
df.isnull().sum()

# Split the data into features (X) and target (y)
X = df.drop(['education'], axis=1)
y = df['education']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Function to encode categorical features
def encode_features(train, test):
    label_encoder = LabelEncoder()
    for column in train.columns:
        if train[column].dtype == 'object':
            train[column] = label_encoder.fit_transform(train[column])
            test[column] = label_encoder.transform(test[column])  # Apply the same encoding to test data
    return train, test

# Encode training and testing features
X_train, X_test = encode_features(X_train, X_test)

# Function to train and evaluate the model
def evaluate_decision_tree(criterion, X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    print(f'\nModel accuracy score with criterion {criterion}: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Training-set accuracy score: {accuracy_score(y_train, y_pred_train):.4f}')
    print(f'Test set score: {clf.score(X_test, y_test):.4f}')

    # Confusion matrix and classification report
    print('\nConfusion matrix\n', confusion_matrix(y_test, y_pred))
    print('\nClassification report\n', classification_report(y_test, y_pred))

    # Plot the decision tree
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, feature_names=X_train.columns, class_names=y_train.unique(), filled=True, rounded=True)
    plt.show()

    # Export the tree to a graphviz format
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X_train.columns,
                                    class_names=y_train.unique(), filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    return graph

# Evaluate Gini Decision Tree
gini_graph = evaluate_decision_tree('gini', X_train, y_train, X_test, y_test)

# Evaluate Entropy Decision Tree
entropy_graph = evaluate_decision_tree('entropy', X_train, y_train, X_test, y_test)

# Show the graphs for both criteria
gini_graph
entropy_graph
