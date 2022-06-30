import pandas as pd
df=pd.read_csv('census_income.csv')

#fixing ? values
df=df.replace('?','')
#columns with nan
cols=['workclass','occupation','native.country']
for col in cols:
    df[col].fillna(df[col].mode()[0],inplace=True)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
for col in df.columns.values:
    if df[col].dtype=='object':
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col])
#feature selection
x=df.drop('income',axis=1)
y=df['income']
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# bestfeatures = SelectKBest(score_func=chi2,k='all')
# bestfeatures.fit(x,y)
# dfscore=pd.DataFrame(bestfeatures.scores_)
# dfcolumns=pd.DataFrame(x.columns)
# featuresScores=pd.concat([dfcolumns,dfscore],axis=1)
# featuresScores.columns=['Features','Score']
# print(featuresScores)

x=x.drop(['workclass','education','race','sex','capital.loss','native.country'], axis=1)

#Scaling features
from sklearn.preprocessing import StandardScaler
for col in x.columns.values:
    scaler=StandardScaler()
    x[col]=scaler.fit_transform(x[col].values.reshape(-1,1))

#Oversampling as there is imbalance in data
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
ros.fit(x,y)
x,y=ros.fit_resample(x,y)

#Model training/testing and results
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB



dt=DecisionTreeClassifier(random_state=0)
rf=RandomForestClassifier(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
lr=LogisticRegression(random_state=0)
sc=svm.SVC(random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print('Decision tree',accuracy_score(y_test,y_pred))

rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print('Random forest',accuracy_score(y_test,y_pred1))

gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print('Gradient Boosting',accuracy_score(y_test,y_pred2))

lr.fit(x_train,y_train)
y_pred3=lr.predict(x_test)
print('Logistic regression',accuracy_score(y_test,y_pred3))

sc.fit(x_train,y_train)
y_pred4=sc.predict(x_test)
print('SVM',accuracy_score(y_test,y_pred4))

gnb.fit(x_train,y_train)
y_pred5=gnb.predict(x_test)
print('Guassian naive bayes',accuracy_score(y_test,y_pred5))


from sklearn.naive_bayes import BernoulliNB
bn=BernoulliNB()
bn.fit(x_train,y_train)
pred=bn.predict(x_test)
print('Bernoullis Naive bayes',accuracy_score(y_test,pred))

#OUTPUT
'''
Decision tree 0.8392857142857143
Random forest 0.8718215441516413
Gradient Boosting 0.8041204345815997
Logistic regression 0.7606622746185853
SVM 0.8142337031900139
Guassian naive bayes 0.6747283865002311
Bernoullis Naive bayes 0.7507801664355063
'''
