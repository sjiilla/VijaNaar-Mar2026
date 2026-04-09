# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 07:10:08 2026

@author: S Jilla
"""

import pandas as pd
import os, io, pydotplus
from sklearn.impute import SimpleImputer #New version
from sklearn import ensemble,tree
from sklearn import model_selection

#changes working directory
os.chdir("C:/Data Science/Data")

titanic_train = pd.read_csv("train.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv('test.csv')
titanic_test.shape
titanic_test.info()
titanic_test.Survived = None

#it gives the same never of levels for all the categorical variables
titanic = pd.concat([titanic_train, titanic_test])

#create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic['Title'] = titanic['Name'].map(extract_title)

#Imputation work for missing data with default values
mean_imputer = SimpleImputer() #By defalut parameter is mean and let it use default one.
mean_imputer.fit(titanic_train[['Age','Fare']]) 

#Age is missing in both train and test data.
#Fare is NOT missing in train data but missing test data. Since we are playing on tatanic union data, we are applying mean imputer on Fare as well..
titanic[['Age','Fare']] = mean_imputer.transform(titanic[['Age','Fare']])

#creaate categorical age column from age
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
titanic['Age1'] = titanic['Age'].map(convert_age)

titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
titanic['FamilySize1'] = titanic['FamilySize'].map(convert_familysize)

#convert categorical columns to one-hot encoded columns
titanic1 = pd.get_dummies(titanic, columns=['Sex','Pclass','Embarked', 'Age1', 'Title', 'FamilySize1'])
titanic1.shape
titanic1.info()

titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
titanic2.shape

#Split Train And Test data
X_train = titanic2[0:titanic_train.shape[0]]
X_train.shape
X_train.info()
y_train = titanic_train['Survived']

#oob scrore is computed as part of model construction process
dt_estimator = tree.DecisionTreeClassifier()
ada_estimator = ensemble.AdaBoostClassifier(estimator = dt_estimator, random_state = 1)
ada_grid = {'n_estimators':[5], 'learning_rate':[0.01,0.02,1.0], 'estimator__max_depth':[3, 4, 6]}
grid_ada_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, cv=10, n_jobs=5)
grid_ada_estimator.fit(X_train, y_train)
print(grid_ada_estimator.cv_results_)
print(grid_ada_estimator.best_score_)
print(grid_ada_estimator.best_params_)
print(grid_ada_estimator.score(X_train, y_train))

#Explore Feature Importances calculated by decision tree algorithm
features = X_train.columns
importances = grid_ada_estimator.best_estimator_.feature_importances_
fe_df = pd.DataFrame({'feature':features, 'importance': importances})
print(fe_df)

X_test = titanic2[titanic_train.shape[0]:]
X_test.shape
X_test.info()
titanic_test['Survived'] = grid_ada_estimator.predict(X_test)

os.getcwd()
titanic_test.to_csv('submissionAdaBoost.csv', columns=['PassengerId','Survived'],index=False)


#extracting all the trees build by random forest algorithm
n_tree = 0
#for est in bag_tree_estimator1.estimator_: 
for est in grid_ada_estimator.best_estimator_:
    dot_data = io.StringIO()
    #tmp = est.tree_
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())#[0] 
    graph.write_pdf("AdaTree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1


