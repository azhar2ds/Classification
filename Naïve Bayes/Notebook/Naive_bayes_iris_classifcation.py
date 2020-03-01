from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
print(dir(iris))
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3,random_state=1)
nb=GaussianNB()
nb.fit(X_train, y_train)
pre=nb.predict(X_test)
print("Accuracy score:", accuracy_score(y_test,pre))
for n in range(150):
    a=nb.predict([X[n]])
    b=y[n]
    if a!=b:
        print("Incorrect Prediction out of 150 total samples")
    
