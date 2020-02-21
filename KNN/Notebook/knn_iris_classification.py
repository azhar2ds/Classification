import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.preprocessing import StandardScalar
from sklearn.preprocessing import StandardScaler


iris=datasets.load_iris()
X=pd.DataFrame(iris.data, columns=iris.feature_names)
y=pd.Series(iris.target)

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.40, random_state=0)

s=StandardScaler()

s.fit(X_train)
X_train=s.transform(X_train)
X_test=s.transform(X_test)

k=KNeighborsClassifier(n_neighbors=8)
k.fit(X_train, y_train)
y_p=k.predict(X_test)
print(accuracy_score(y_p,y_test))

from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors = 10)
knnr.fit(X, y)