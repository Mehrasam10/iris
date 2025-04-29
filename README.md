# iris
# Library Importation
pd is imported as the pandas library.
We import NumPy library and refer it as np.
Load the iris dataset from sklearn.datasets module.
from sklearn.model_selection import train_test_split, cross_val_score
Loading the RandomForestClassifier from sklearn.ensemble module.
Import the SVC class from sklearn.svm module.
Importing KNeighborsClassifier Module from the sklearn.neighbors Library.
Importing the LogisticRegression class from the sklearn.linear_model.
Get import accuracy_score, classification_report and confusion_matrix from sklearn.metrics module.
Import the matplotlib.pyplot module and assign it to an alias of plt.
In this case we import the Seaborn library using the alias 'sns'.

# Import the Dataset
Function Definition: load_data()
iris_dataset = load_iris()
The 'iris.data' array is used to create a DataFrame X with column headers specified by 'iris.feature_names'.
A pandas Series with the Series labeled as 'species' is assigned to the variable y which is constructed from the target attribute of the iris dataset.
The iris.target_names is returned, as well as X and y.

Model Training and Evaluation
def assess_model_performance(X_train, X_test, y_train, y_test):
models = {
Random Forest: RandomForestClassifier with a random state parameter set to 42.
Linear Kernel Support Vector Classifier (SVC) Support Vector Machine (SVM)
The identifier for KNeighborsClassifier is 'KNN'.
To implement logistic regression, the LogisticRegression model was used and maximum number of iterations was set to 200.
    }
We initialize an empty collection to be a dictionary named 'scores'.

Get the key as 'name' and its value as 'model' for each entry in the models dictionary.
Then the model is trained on the given training data X_train and y_train.
The model is made to predict on X_test, and the predictions are assigned to y_pred.
The value produced by the accuracy_score function by comparing the true labels(y_test) and predicted labels(y_pred) is assigned to the variable 'acc'.
For the specified 'name', the variable 'scores' is set to the value of 'acc'.
Show the following message: "--- {name}--- "
print("Accuracy:", acc)
Print the classification report for y_test and y_pred.

Return the scores.

Identifying the Optimal Model
def determine_optimal_model_name(model_scores):
    Return element with highest value in scores

Visualization of Feature Importance in Random Forest Models
def display_feature_significance(X, y):
We instantiated the RandomForestClassifier with fixed random state value of 42.
The fit method trains the random forest model on the data X and its corresponding target values y.
The random forest model generates the feature importances and the variable 'importances' is assigned to it.
We create a DataFrame named importance_df with two columns: 'Feature' being the names of the features from X.columns and 'Importance' with the corresponding importance values from importances.
We sort the variable 'importance_df' in descending order according to 'Importance' column.

Use the appropriate plotting function to set the figure dimensions to 8 units in width, and 5 units in height.
Using the data from 'importance_df' we generated a bar plot using Seaborn to visualize feature importance and plotted 'Importance' on the x-axis and 'Feature' on the y-axis, using 'viridis' color palette.
Random Forest Feature Importance Visualization
    plt.tight_layout()
    plt.show()

Bar Chart Comparing Accuracy
def display_accuracy_comparison(scores):
We assign a list with all the keys of the 'scores' dictionary to the variable 'names'.
The list is assigned to the variable 'values', which is populated with the elements taken from the values of the 'scores' dictionary.
Use the appropriate plotting function to set the figure dimensions to 8 units in width, and 5 units in height.
It uses the 'sns.barplot' function with the 'names' for the x axis, 'values' for the y axis and the 'Set2' color palette.
The chart title 'Comparison of Model Accuracy' is set.
    plt.ylabel('Accuracy')
Make the y axis limits be from 0.8 to 1.0.
    plt.tight_layout()
    plt.show()

Primary Execution Program
Primary Function Definition
For load_data(), the values are assigned to variables X, y, and class_names.
The dataset was partitioned into training and testing by using a fixed random seed (random_state=42) such that 80% of the data would be used for training and 20% for testing.

Showing Evaluation Results for Multiple Models
Here we assign the results, which are the result of assessing the models using training and test datasets namely X_train, X_test, y_train, and y_test, to the variable scores.

Execute print("\nBest Model:", best_model_name(scores)) to display the most optimal model.

The Feature Importance Visualization using the Random Forest Algorithm...
Plot the feature importance using the variables X and y.

Compare the Model Accuracy Results
    plot_accuracy_comparison(scores)

It is executed if script is being executed directly as main program.
    main()
