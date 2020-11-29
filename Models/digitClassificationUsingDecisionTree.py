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

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

classifier.fit(pca_train,y_train)
y_pred = classifier.predict(pca_test)

from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test,y_pred)
mat

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))