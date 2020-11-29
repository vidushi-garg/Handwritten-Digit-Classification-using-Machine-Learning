import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
dataset = pd.read_csv('train.csv')

X = dataset.iloc[:,1:]
y = dataset['label']

# Splittng the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)

# Since  the data conatins many columns so applying pca
from sklearn.decomposition import PCA
pca = PCA(n_components = 36)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)

var = pca.explained_variance_ratio_

from sklearn.model_selection import GridSearchCV
params =  {
            "n_estimators" : [10,50,100],
            "max_features" : ["auto", "log2", "sqrt"],
            "bootstrap"    : [True, False]
        }
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

clas = GridSearchCV(classifier, params, n_jobs=-1, cv=5, verbose=1, scoring='accuracy')
clas.fit(pca_train, y_train)

clas.best_estimator_

classifier = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features='log2', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(y_test,y_pred))

