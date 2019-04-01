# Classify recipes into regional cuisines based on ingredients, using logistic regresion, SVM, randomforest, MultinomialNB
# Visualize best model results with a confusion matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore") 

def logistic_test(X,y):
    """
    Inputs: X is the matrix of explanatory variables, y is the vector of target variables
    Function: Fits a logistic regression. Tunes parameters. Prints out the accuracy for best parameter. Pickles the model with the best accuracy result.	
    Outputs: None
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    crange = [0.01,0.1,1,10,100] # Tuning parameters
    acc_scr = []
	
    for num in crange:
        model = LogisticRegression(C=num,solver='liblinear',max_iter = 5000,multi_class='auto')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mas = metrics.accuracy_score(y_test,y_pred)
        acc_scr.append(mas)
	
    max_idx = max(range(len(acc_scr)), key=acc_scr.__getitem__)
    best_model = LogisticRegression(C=crange[max_idx],solver='liblinear',max_iter = 5000,multi_class='auto')
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print ('Logistic Test Accuracy: ', metrics.accuracy_score(y_test,y_pred))
	
    with open('model/logistic.pickle', 'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
    return None

def svm_test(X,y):
    """
    Inputs: X is the matrix of explanatory variables, y is the vector of target variables
    Function: Fits a Support Vector Machine learning model. Tunes parameters. Prints out the accuracy for best parameter. Pickles the model with the best accuracy result.	
    Outputs: None
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
    crange = [0.01,0.1,1,10,100] # Tuning parameters
    acc_scr = []
    
    for num in crange:
        model = svm.LinearSVC(C=num)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mas = metrics.accuracy_score(y_test,y_pred)
        acc_scr.append(mas)

    max_idx = max(range(len(acc_scr)), key=acc_scr.__getitem__)
    best_model = svm.LinearSVC(C=num)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print ('SVM Test Accuracy: ', metrics.accuracy_score(y_test,y_pred))

    with open('model/svm.pickle', 'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
    return None

def nb_test(X,y):
    """
    Inputs: X is the matrix of explanatory variables, y is the vector of target variables
    Function: Fits a Naive Bayes model. Pickles the model. Prints accuracy result.	
    Outputs: None
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print ('NB Test Accuracy: ', metrics.accuracy_score(y_test,y_pred))

    with open('model/nb.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
    return None

def rf_test(X,y):
    """
    Inputs: X is the matrix of explanatory variables, y is the vector of target variables
    Function: Fits a Naive Bayes model. Pickles the model. Prints accuracy result.	
    Outputs: None
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    rf_model = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print ('Random Forest Test Accuracy: ',metrics.accuracy_score(y_test,y_pred))

    with open('model/rf.pickle', 'wb') as handle:
        pickle.dump(rf_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
    return None

def plot_confusion_matrix(cm, col, title, cmap=plt.cm.viridis):
    """
    Inputs: cm is the columns, col is the target, title is used for plot, cmap is the color scheme (default set as ocean)
    Function: Plots the confusion matrix
    Outputs: None 
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    for i in range(cm.shape[0]):
        plt.annotate("%.2f" %cm[i][i],xy=(i,i),
                    horizontalalignment='center',
                    verticalalignment='center')
    plt.title(title,fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(col.unique()))
    plt.xticks(tick_marks, sorted(col.unique()),rotation=90)
    plt.yticks(tick_marks, sorted(col.unique()))
    plt.tight_layout()
    plt.ylabel('True label',fontsize=10, labelpad=-10)
    plt.xlabel('Predicted label',fontsize=10)

if __name__ == '__main__':
    yum_clean = pd.read_pickle('data/yummly_clean.pkl')

    #Find the set of ingredients used
    yum_ingredients = set()
    yum_clean['clean ingredients'].map(lambda x: [yum_ingredients.add(i) for i in x])
    yum = yum_clean.copy()
    for item in yum_ingredients:
        yum[item] = yum['clean ingredients'].apply(lambda x:item in x)
    yum_X = yum.drop(yum_clean.columns,axis=1)

    #Test for various classification models
    logistic_test(yum_X, yum['cuisine'])
    svm_test(yum_X,yum['cuisine'])
    nb_test(yum_X,yum['cuisine'])
    rf_test(yum_X,yum['cuisine'])

    #Plot confusion matrix with Random Forest model
    X = yum_X.values
    y = yum['cuisine']
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    model = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,12))
    plot_confusion_matrix(cm_normalized, yum['cuisine'],title='Percentage Correctly Identified Cuisines using only Ingredients as a Predictor')
    plt.savefig('img/confusionmatrix.jpg')
