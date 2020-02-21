import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.preprocessing import StandardScalar
from sklearn.preprocessing import StandardScaler


iris=datasets.load_iris()
X=iris.data
y=iris.target
names=iris.feature_names

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.40, random_state=0)


s=StandardScaler()
s.fit(X_train)
X_train=s.transform(X_train)
X_test=s.transform(X_test)

print('done')
print(X_train.shape)
print(X_test.shape)
print(type(X_train))
print(type(X_test))

k=KNeighborsRegressor(n_neighbors=3)
k.fit(X,y)
p=k.predict(X_test)
print(accuracy_score(p,y_test))
