import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
d=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
#headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
#d=pd.read_csv(url, names=headernames)
#print(d.head())
X=d.iloc[:,:-1].values
y=d.iloc[:,3].values


print(y)
print(len(y))
print(y.shape)



X = d.iloc[:, :-1].values
y = d.iloc[:, 4].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)



scaler = StandardScaler()
scaler.fit(X_train)
print('mean of each column',scaler.mean_)
print('STD of each column',scaler.std_)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)



result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
