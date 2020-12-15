# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:09:26 2020

@author: Study
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:10:26 2020

@author: Study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score


def calc_precision_recall(confusion_matrix):
    # Calc precision and relall curve#
    #disp = plot_precision_recall_curve(tree_clf, features_test, target_test)
    tn = (confusion_matrix[0,0])
    fn = (confusion_matrix[1,0])
    fp = (confusion_matrix[0,1])
    tp = (confusion_matrix[1,1])
    
    precision = tp/(tp + fp)
    recall = tp/(tp+fn)
    return precision, recall

def calc_roc(confusion_matrix):
    tn = (confusion_matrix[0,0])
    fn = (confusion_matrix[1,0])
    fp = (confusion_matrix[0,1])
    tp = (confusion_matrix[1,1])
    
    tpr = tp/(tp+fn)
    fpr = fp/(tn + fp)
    return tpr, fpr
#%% 
def generate_confusion_matrix(target_test, target_predicted):
    
    #Create Confusion Matrix
    matrix = confusion_matrix(target_test, target_predicted)
    class_names = ['Churn_No', 'Churn_Yes']
    df_confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)

    # Heatmap
    sns.heatmap(df_confusion, annot=True, cmap="Blues", fmt=".0f")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig('./confusion_matric_num.png')
    plt.show()
    plt.close()

    # calc precision and recall
    precision, recall = calc_precision_recall(matrix)
    return df_confusion, precision, recall


#%% Data Preprocessing
# Graphics output

print("\n====================================================================")
print("BEGIN")
print("====================================================================\n")
    
def visualise_correlations(df, target_var):
    # Convert dummy variables
    df.head()
    
    # Replace churn string values with numeric binary values
    # Utilize pandas dummy variable function to create dummy variable series for categorical data
    dummy_df = pd.get_dummies(df)
    dummy_df.info()
    #dummy_df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)
    
    print(dummy_df.head())
    return dummy_df
    

def get_data_telco(colnames, dummy_cols, filename):
    print("\n--------------------------------------------------------------------")
    print("Import Data\n")


    # Import dataset
    df = pd.read_csv(filename)
    
    # Remove records with N/A values
    df.dropna(inplace=True)
    
    # Select attributes to be included in ff
    df_reduced = df[colnames]
      
    # Create Feature Frame and Target Frame
    ff = df_reduced.iloc[:,1:-1] # All columns except target column
    tf =  df_reduced.iloc[:, -1] # Only the last column
    
    # Do one-hot-encoding by listing columns ( Create dummy variables)
    
    feature_frame = pd.get_dummies(ff, columns=dummy_cols, drop_first=True)
    feature_names= feature_frame.columns
    features = pd.DataFrame(feature_frame).to_numpy()
    
    target_frame = pd.get_dummies(tf, columns=dummy_cols, drop_first=True)
    target_name=target_frame.columns
    target = pd.DataFrame(target_frame).to_numpy()
    target = np.ravel(target)
    
    # Split into training and test set
    features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=42)

    return features_train, features_test, target_train, target_test, feature_frame, target_frame, df

def tree_clf_model(features_train, features_test, target_train, target_test, ff, tf):
    #%% Set model parameters
    print("\n--------------------------------------------------------------------")
    print("Decision Tree Classifier\n")
    
    plot_tree_yes = 1
    tree_depth = 10
    min_in_leaf = 5
    classifier = DecisionTreeClassifier(max_features="auto", random_state=42, min_samples_leaf=min_in_leaf, max_depth=tree_depth)
    
    # Train model
    tree_clf = classifier.fit(features_train, target_train)
    
    # Export output tree
    r = export_text(tree_clf, show_weights=True, feature_names=list(ff.columns))
    
    # Make Predictions
    target_predicted = tree_clf.predict(features_test)
    
    df_confusion, precision, recall = generate_confusion_matrix(target_test, target_predicted)
    
    print(precision_score(target_test, target_predicted))
    print(recall_score(target_test, target_predicted))
    
    # Summarise model
    print("Tree Depth: ", tree_depth)
    print("Minimum in Leaf: ", min_in_leaf)
    
    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
    
    if plot_tree_yes == 1:
        plot_tree(tree_clf, filled=True)
        plt.show()
    
    
    return tree_clf, r, df_confusion

def grid_search_tree():
    # Use grid search to find good hyperparameter values
    #print("Grid Search Cross Validation:")
    #params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split':[2,3,4]}
    #grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
    #grid_search_cv.fit(features_train, target_train)
    #print(grid_search_cv.best_estimator_)
    print("Grid Search:")


#%% Random Foret Classifier
def rnd_clf_model(features_train, features_test, target_train, target_test):
    print("\n--------------------------------------------------------------------")
    print("Random Forest Classifier\n")
    
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rnd_clf.fit(features_train, target_train)
    
    # Make Predictions
    target_predicted = rnd_clf.predict(features_test)
    
    # Confusion Matrix
    df_confusion, precision, recall = generate_confusion_matrix(target_test, target_predicted)
    
    # Summarise model
    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
    
    if plot_tree_yes == 1:
        plot_tree(rnd_clf, filled=True)
        plt.show()

    #%% MAIN FUNCTION:
    #============================================================================================================================================

# Analysis Parameters:
plot_tree_yes = 1
filename = "data.csv"

# Define categorical attributes that need dummy features created
dummy_cols = ["gender", "Contract"]

'''#dummy_cols = ["gender", 
                  "Partner", 
                  "Dependents", 
                  "PhoneService", 
                  "MultipleLines", 
                  "InternetService", 
                  "OnlineSecurity", 
                  "OnlineBackup", 
                  "DeviceProtection", 
                  "TechSupport", 
                  "StreamingTV", 
                  "StreamingMovies", 
                  "Contract", 
                  "PaperlessBilling",
                  "PaymentMethod"]'''

# Specify attributes that we wish to use in the analysis
colnames = ["customerID", "gender", "tenure", "Contract", "MonthlyCharges", "TotalCharges", "Churn"]

# Get data, prepare, extract features, split into reaining and test sets
features_train, features_test, target_train, target_test, feature_frame, target_frame, df = get_data_telco(colnames, dummy_cols, filename)

# Train decision tree model, and print evaluation metrics
tree_clf, tree_output, df_confusion = tree_clf_model(features_train, features_test, target_train, target_test, feature_frame, target_frame)
# rnd_clf_model(features_train, features_test, target_train, target_test)


