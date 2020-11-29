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
params = {"C":[0.1,1.0,10]
          }
from sklearn.svm import SVC
classifier = SVC()

clas = GridSearchCV(classifier, params, n_jobs=-1, cv=5, verbose=1, scoring='accuracy')
clas.fit(pca_train, y_train)

clas.best_estimator_


classifier = SVC(C=30, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto',
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.01, verbose=False)

classifier.fit(pca_train,y_train)
y_pred = classifier.predict(pca_test)

from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(y_test,y_pred))
