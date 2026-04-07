# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 07:05:24 2026

@author: S Jilla
"""

import os
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Data Science\\Data")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#Random Forest classifier
#Remember RandomForest works for Decission Trees only and there is NO Base_Estimator parameter exists
rf_estimator = ensemble.RandomForestClassifier(random_state=1)
#n_estimators: no.of trees to be built
#max_features: Maximum no. of features to try with
#rf_grid = {'n_estimators':list(range(200,251,50)),'max_features':[3,6,9],'criterion':['entropy','gini']}
rf_grid = {'n_estimators':list(range(250,551,50)),'max_features':[11],'criterion':['entropy','gini']}
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator,rf_grid, cv=10, n_jobs=10)
rf_grid_estimator.fit(X_train, y_train)
#rf_grid_estimator.grid_scores_
rf_grid_estimator.best_estimator_
rf_grid_estimator.best_score_
rf_grid_estimator.score(X_train, y_train)

#Feature Importance
fi_df = pd.DataFrame({'feature':X_train.columns, 'importance':  rf_grid_estimator.best_estimator_.feature_importances_}) #You may notice that feature	importance "Title_Mr" has more importance
print(fi_df)

titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = rf_grid_estimator.predict(X_test)
titanic_test.to_csv("submission_rf.csv", columns=['PassengerId','Survived'], index=False)
os.getcwd()

#Print #extracting all the trees build by random forest algorithm
n_tree = 0
#for est in bag_tree_estimator1.estimator_: 
for est in rf_grid_estimator.best_estimator_:
    dot_data = io.StringIO()
    #tmp = est.tree_
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())#[0] 
    graph.write_pdf("RFTree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1

