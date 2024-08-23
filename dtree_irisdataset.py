# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:26:52 2018

@author: sidhidatrinayak
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split



#iris data 
iris = load_iris()
iris
iris.data[0]
iris.target_names

import pandas as pd

#perform test aND SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4)
df2=pd.DataFrame(X_train,y_train)
df2
clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=10)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

df1=pd.DataFrame(y_test,y_pred)
df1
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))

tree.plot_tree(clf)

