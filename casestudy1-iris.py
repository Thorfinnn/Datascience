import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("IRIS.csv")

lreg = LogisticRegression(random_state=0)
rf=RandomForestClassifier(random_state=0)
dc=DecisionTreeClassifier(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
sv=svm.SVC()
mlp=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()

x = df.drop("species", axis=1)
y = df["species"]
# print(x)
# print(y)
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)
lreg.fit(x_train, y_train)
y_pred0=lreg.predict(x_test)


sv.fit(x_train, y_train)
y_pred1=sv.predict(x_test)


dc.fit(x_train, y_train)
y_pred2=dc.predict(x_test)


rf.fit(x_train, y_train)
y_pred3=rf.predict(x_test)


gb.fit(x_train, y_train)
y_pred4=gb.predict(x_test)


mlp.fit(x_train, y_train)
y_pred5=mlp.predict(x_test)

gnb.fit(x_train, y_train)
y_pred6=gnb.predict(x_test)

mnb.fit(x_train, y_train)
y_pred7=mnb.predict(x_test)
print("Logistic regresion",accuracy_score(y_test, y_pred0))
print("svm",accuracy_score(y_test, y_pred1))
print("Decision tree",accuracy_score(y_test, y_pred2))
print("Random Forest",accuracy_score(y_test, y_pred3))
print("Gradient Boosting",accuracy_score(y_test, y_pred4))
print("Nueral networks",accuracy_score(y_test, y_pred5))
print("Guassian Naive Bayes",accuracy_score(y_test, y_pred6))
print("Multiomial Naives bayes",accuracy_score(y_test, y_pred7))

# Output
# Logistic regresion 0.9047619047619048
# svm 0.8857142857142857
# Decision tree 0.9428571428571428
# Random Forest 0.9619047619047619
# Gradient Boosting 0.9523809523809523
# Nueral networks 0.3142857142857143
# Guassian Naive Bayes 0.9333333333333333
# Multiomial Naives bayes 0.638095238095238
