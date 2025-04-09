# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:32:22 2023

@author: stimp
"""

#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

#loading data
credit_data = pd.read_csv("C:/Users/stimp/OneDrive/Desktop/Flame/OPSM322/Pulkit_Pincha_OPSM322_Homework-2/Questions/credit_h new.csv")
credit_data.head()

credit_features = credit_data.iloc[:,0:8]  
credit_labels = credit_data['R1']

#split into train and test 
x_train, x_test, y_train, y_test = train_test_split(credit_features, credit_labels, test_size = 0.20, random_state = 23)

#KNN
KNNclassifier = KNeighborsClassifier(n_neighbors=5)
KNNmodel = KNNclassifier.fit(x_train, y_train)
KNNpreds = KNNmodel.predict(x_test)
print("Accuracy KNN:", accuracy_score(y_test, KNNpreds) *100)

#LDA
LDAmodel = LinearDiscriminantAnalysis()
LDAmodel.fit(x_train, y_train)
LDApreds = LDAmodel.predict(x_test)
print("Accuracy Linear DA:", accuracy_score(y_test, LDApreds) *100)

#QDA
QDAmodel = QuadraticDiscriminantAnalysis()
QDAmodel.fit(x_train, y_train)
QDApreds = QDAmodel.predict(x_test)
print("Accuracy Quadratic DA:", accuracy_score(y_test, QDApreds) *100)

#SVM
SVMmodel = SVC(kernel='linear')
SVMmodel.fit(x_train, y_train)
SVMpreds = SVMmodel.predict(x_test)
print("Accuracy SVM:", accuracy_score(y_test, SVMpreds) *100)

#DT
DTmodel = DecisionTreeClassifier(max_depth=4)
DTmodel.fit(x_train, y_train)
DTpreds = DTmodel.predict(x_test)
print("Accuracy DT:", accuracy_score(y_test, DTpreds) *100)

#NB
NBmodel = GaussianNB()
NBmodel.fit(x_train, y_train)
NBpreds = NBmodel.predict(x_test)
print("Accuracy NB:", accuracy_score(y_test, NBpreds) *100)

#cross-validation
models = [KNNmodel, LDAmodel, QDAmodel, SVMmodel, DTmodel, NBmodel]
modelnames = ['KNN Model', 'LDA Model', 'QDA Model', 'SVM Model', 'DT Model', 'NB Model']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model, name in zip(models, modelnames):
    cvresults = cross_validate(model, x_train, y_train, cv=kf, scoring='accuracy')
    bestmodelindex = cvresults['test_score'].argmax()
    bestmodel = model
    bestmodel.fit(x_train, y_train)
    testaccuracy = (accuracy_score(y_test, bestmodel.predict(x_test))) *100
    print(f"Cross-Validation Accuracy of {name}: {cvresults['test_score'][bestmodelindex] *100}")
    print(f"Test Accuracy of {name}: {testaccuracy}")

##GridSearchCV
#KNN
KNNparamgrid = {'n_neighbors': [2, 4, 6, 8, 10]}
KNNgrid = GridSearchCV(KNeighborsClassifier(), KNNparamgrid, cv=5)
KNNgrid.fit(x_train, y_train)
bestKNNmodel = KNNgrid.best_estimator_
KNNtestaccuracy = (accuracy_score(y_test, bestKNNmodel.predict(x_test))) *100

#DT
DTparamgrid = {'max_depth': [2, 4, 6, 8]}
DTgrid = GridSearchCV(DecisionTreeClassifier(), DTparamgrid, cv=5)
DTgrid.fit(x_train, y_train)
bestDTmodel = DTgrid.best_estimator_
DTtestaccuracy = (accuracy_score(y_test, bestDTmodel.predict(x_test))) *100

print("Best KNN Model Test Accuracy:", KNNtestaccuracy)
print("Best Decision Tree Model Test Accuracy:", DTtestaccuracy)