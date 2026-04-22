# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:16:31 2026

@author: S Jilla
"""

import pandas as pd, time
from sklearn import tree, decomposition

#from sklearn.tree import plot_tree
#import matplotlib.pyplot as plt

titanic_train = pd.read_csv("C:/Data Science/Data/train.csv")

#EDA
titanic_train.shape #Gives the counts rows and columns
titanic_train.info() #Gives the statistics information
titanic_train.describe() #Nullable, Type of columns Data
#Transformation of non numneric cloumns to 1-Hot Encoded columns
#There is an exception with the Pclass. Though it's co-incidentally a number column but it's a Categoric column(Even common-sence wise).

#Transform categoric to One hot encoding using get_dummies
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.describe()
titanic_train1.head(10)

#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], axis=1, inplace=False)
X_train.info()
X_train.shape
y_train = titanic_train['Survived']
X_train.info()

#PCA work
pca = decomposition.PCA(n_components=3)
pca.fit(X_train)
#Transformation of PCA happens here
transformed_X_train = pca.transform(X_train)

start_time = time.perf_counter()

dt = tree.DecisionTreeClassifier()
#.fit builds the model. In this case the model building is using Decission Treee Algorithm
dt.fit(transformed_X_train,y_train)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(execution_time)

#predict the outcome using decission tree
titanic_test = pd.read_csv("C:/Data Science/Data/test.csv")
titanic_test.shape
#Fill missing data of Test(Fare)
titanic_test.info() #Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
#Data Imputation
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean() #Older version
#titanic_test.loc[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean() #New Version

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], axis=1, inplace=False)
#Apply the model on future/test data

#PCA on Test data
pca = decomposition.PCA(n_components=3)
pca.fit(X_test)
X_transformed_Test = pca.transform(X_test)

titanic_test['Survived'] = dt.predict(X_transformed_Test)
titanic_test.to_csv("Submission_PCA3.csv", columns=['PassengerId', 'Survived'], index=False)
