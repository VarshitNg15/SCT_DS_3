Decision Tree Classifier Project

Project Overview

This project aims to implement a Decision Tree Classifier to predict the education level of individuals using a dataset from a banking campaign. The dataset is processed using Pandas for data manipulation and scikit-learn for machine learning model creation. The code compares the performance of decision tree models using two criteria: Gini Impurity and Entropy.

Files

1. task3.py — Main Python script implementing the decision tree classifier.
2. bank-full.csv — Dataset file used for classification.

Code Overview

1. Data Loading and Preprocessing:
    - The dataset is loaded using Pandas.
    - The column names are manually defined for the dataset.
    - Missing values are checked (if any).
    - Features (X) and target (y) are separated.
    - The dataset is split into training and testing sets (67% train, 33% test) using train_test_split().

2. Feature Encoding:
    - Categorical features are label-encoded using LabelEncoder from scikit-learn.
    - The encoding is applied to both training and testing sets consistently.

3. Model Training and Evaluation:
    - Two Decision Tree models are trained using different criteria:
        - Gini Impurity: Measures the impurity of a node.
        - Entropy: Measures the information gain.
    - The models are evaluated based on:
        - Accuracy score on the test set
        - Confusion matrix and classification report
    - The decision tree structure is plotted using matplotlib and graphviz.

4. Results:
    - The accuracy, confusion matrix, and classification reports are printed for both models.
    - Decision tree graphs are generated to visualize the structure of the models.

Output

- The model's accuracy on the training and testing sets.
- Confusion matrix and classification report for both models (Gini and Entropy).
- Graphical visualization of the decision trees.

Model Accuracy

- Gini Model: Outputs accuracy and evaluation metrics based on Gini impurity.
- Entropy Model: Outputs accuracy and evaluation metrics based on entropy.

Conclusion

This project provides an implementation of a decision tree classifier using two different impurity measures. The results can be compared to understand how each criterion affects the performance of the model.
