import pandas as pd
df=pd.read_csv('winequalityN.csv')
# print(df.describe())
x=df.drop('quality',axis=1)
y=df['quality']
for col in x.columns.values:
    if df[col].count()==6497:
        continue
    df[col].fillna(df[col].mode()[0],inplace=True)
# print(df.describe())
for col in ['density','alcohol','quality']:
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    r=q3-q1
    upper=q3+1.5*r
    lower=q1-1.5*r
    o1=df[df[col]<lower].values
    o2=df[df[col]>upper].values
    df[col].replace(o1,lower,inplace=True)
    df[col].replace(o2,upper,inplace=True)
x=df.drop('quality',axis=1)
y=df['quality']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(x.type)
x.type=le.transform(x.type)

# from matplotlib import pyplot as plt
# from sklearn.ensemble import ExtraTreesClassifier
# import seaborn as sns
# et = ExtraTreesClassifier()
# et.fit(x,y)
# print(et.feature_importances_)
#
# feat_imp=pd.Series(et.feature_importances_,index=x.columns)
# feat_imp.nlargest(11).plot(kind='barh')
# plt.show()
#
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



x=x.drop('residual sugar',axis=1)
x=x.drop('free sulfur dioxide',axis=1)
x=x.drop('pH',axis=1)


from imblearn.over_sampling import SMOTE
smote=SMOTE(k_neighbors=4)
smote.fit(x,y)
x,y=smote.fit_resample(x,y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


dt=DecisionTreeClassifier(random_state=0)
rf=RandomForestClassifier(random_state=0)
gb=GradientBoostingClassifier()
lr=LogisticRegression(random_state=0)
sc=svm.SVC(random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, test_size=0.2)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print('Decision Tree',accuracy_score(y_test,y_pred))

rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print('Random Forest',accuracy_score(y_test,y_pred1))

gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print('Gradient Boosting',accuracy_score(y_test,y_pred2))

#Output
'''
Decision Tree 0.8076051372450265
Random Forest 0.8849156383782423
Gradeint Boosting 0.7275245530093175
'''
