import pandas as pd
df=pd.read_csv('winequalityN.csv')
print(df.describe())
x=df.drop('type',axis=1)
y=df['type']
for col in x.columns.values:
    if df[col].count()==6497:
        continue
    df[col].fillna(df[col].mean(),inplace=True)
print(df.describe())
x=df.drop('type',axis=1)
y=df['type']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y=le.transform(y)

from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
et = ExtraTreesClassifier()
et.fit(x,y)
print(et.feature_importances_)

feat_imp=pd.Series(et.feature_importances_,index=x.columns)
feat_imp.nlargest(11).plot(kind='barh')
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2,k='all')
bestfeatures.fit(x,y)
dfscore=pd.DataFrame(bestfeatures.scores_)
dfcolumns=pd.DataFrame(x.columns)
featuresScores=pd.concat([dfcolumns,dfscore],axis=1)
featuresScores.columns=['Features','Score']
print(featuresScores)


for col in ['density','alcohol','quality']:
    q1=x[col].quantile(0.25)
    q3=x[col].quantile(0.75)
    r=q3-q1
    upper=q3+1.5*r
    lower=q1-1.5*r
    o1=x[x[col]<lower].values
    o2=x[x[col]>upper].values
    x[col].replace(o1,lower,inplace=True)
    x[col].replace(o2,upper,inplace=True)
x=x.drop('density',axis=1)
x=x.drop('alcohol',axis=1)
x=x.drop('pH',axis=1)
# x=x.drop('citric acid',axis=1)
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
print(accuracy_score(y_test,y_pred))

rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print(accuracy_score(y_test,y_pred1))

gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print(accuracy_score(y_test,y_pred2))

lr.fit(x_train,y_train)
y_pred3=lr.predict(x_test)
print(accuracy_score(y_test,y_pred3))

sc.fit(x_train,y_train)
y_pred4=sc.predict(x_test)
print(accuracy_score(y_test,y_pred4))

gnb.fit(x_train,y_train)
y_pred5=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred5))

mnb.fit(x_train,y_train)
y_pred=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred))

from sklearn.naive_bayes import BernoulliNB
bn=BernoulliNB()
bn.fit(x_train,y_train)
pred=bn.predict(x_test)
print(accuracy_score(y_test,pred))
#
# 0.9753737906772207
# 0.9890061565523307
# 0.9736147757255936
# 0.9727352682497801
# 0.9278803869832893
# 0.9577836411609498
# 0.9377748460861918
# 0.7755057167985928

