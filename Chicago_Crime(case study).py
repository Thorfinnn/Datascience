'''
In this case study we are trying to predict weather culprit
Will be arrested or not given case file consisting of many features
Based on historical data
'''
import math

import numpy as np
import pandas as pd
df1=pd.read_csv("C:/Users/hp/Desktop/Chicago Crime/Chicago_Crimes_2001_to_2004.csv",error_bad_lines=False)
df2=pd.read_csv("C:/Users/hp/Desktop/Chicago Crime/Chicago_Crimes_2005_to_2007.csv",error_bad_lines=False)
df3=pd.read_csv("C:/Users/hp/Desktop/Chicago Crime/Chicago_Crimes_2008_to_2011.csv",error_bad_lines=False)
df4=pd.read_csv("C:/Users/hp/Desktop/Chicago Crime/Chicago_Crimes_2012_to_2017.csv",error_bad_lines=False)

df=pd.concat([df1,df2,df3,df4],ignore_index=False,axis=0)
print(df.shape)
df.drop_duplicates(subset=['ID','Case Number'],inplace=True)
print(df.shape)
df.dropna()
print(df.shape)
# print(df.info())
# print(df.head)
# df.to_csv('processed.csv')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Block']=le.fit_transform(df['Block'])
df['Primary Type']=le.fit_transform(df['Primary Type'])
df['Location Description']=le.fit_transform(df['Location Description'])
df['Description']=le.fit_transform(df['Description'])

# from matplotlib import pyplot as plt
# df.groupby([df.Year]).size().plot(kind='barh')
# plt.ylabel('Year')
# plt.xlabel('No of crimes')
# plt.show
# df=df.sample(1000000)
print(df.info())
# df.groupby([df.dayOfWeek]).size().plot(kind='barh')
# plt.xlabel('No of crimes')
# plt.show
# df=pd.read_csv('processed1.csv')
print(df.shape)
df=df.dropna()

#removing points with no x,y coordinates
df[['X Coordinate','Y Coordinate']]=df[['X Coordinate','Y Coordinate']].replace(0.0,np.nan)
df=df.dropna()
# df=df.drop(['Unnamed: 0.2','Unnamed: 0.1','Unnamed: 0'],axis=1)

#Chicago city is bounded by box 41.6439,-87.9401;41,9437,-87.58782
df['Latitude']=df['Latitude'].astype('float64')
df=df[(((df.Latitude>=41.64)&(df.Longitude<=-87.50))|
       ((df.Latitude<=41.94)&(df.Longitude>=-81.94)))]



#removing records with mismatch year
# df=df[df['Date'].year==df.Year]
# df=df.sample(n=100000)
# df.index=pd.DatetimeIndex(df.Date)
# df['Month']=df.index.month
# df['dayOfWeek']=df.index.dayofweek
# df['dayOfMonth']=df.index.day
# df['dayOfYear']=df.index.dayofyear
# df['weekOfMonth']=df.dayOfMonth.apply(lambda d:(d-1)//7+1)
# dayOfYear=list(df.index.dayofyear)
# weekOfYear=[math.ceil(i/7) for i in dayOfYear]
# df['weekOfYear']=weekOfYear
# print(df)
# df=df.drop('Unnamed: 0',axis=1)
df=df.drop('ID',axis=1)
df=df.drop('Case Number',axis=1)
df=df.drop('Date',axis=1)
# df=df.drop('Date.1',axis=1)
df=df.drop('IUCR',axis=1)
# df=df.drop('Description',axis=1)
df=df.drop('FBI Code',axis=1)
df=df.drop('Updated On',axis=1)
df=df.drop('Location',axis=1)

print(df.info())
print(df['Location Description'].unique())

# df.to_csv('processed1.csv')

print(df.info())
x=df.drop('Arrest',axis=1)
y=df['Arrest']
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
#
# et = ExtraTreesClassifier()
# et.fit(x,y)
# print(et.feature_importances_)
# cor=df.corr()
# print(abs(cor['Arrest']))
# feat_imp=pd.Series(et.feature_importances_,index=x.columns)
# feat_imp.nlargest(20).plot(kind='barh')
# plt.show()
# plt.figure(figsize=(20,10))
x=x.drop('Block',axis=1)
x=x.drop('Unnamed: 0',axis=1)
x=x.drop('District',axis=1)
x=x.drop('Ward',axis=1)
x=x.drop('Community Area',axis=1)
x=x.drop('X Coordinate',axis=1)
x=x.drop('Y Coordinate',axis=1)
x=x.drop('Year',axis=1)
# x=x.drop('Month',axis=1)
# x=x.drop('dayOfWeek',axis=1)
# x=x.drop('dayOfMonth',axis=1)
# x=x.drop('dayOfYear',axis=1)
# x=x.drop('weekOfMonth',axis=1)
print(x.info())
# sns.heatmap(cor,annot=True)
# plt.show()
x=x.drop('Beat',axis=1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


mlp=MLPClassifier(hidden_layer_sizes=(6,2),solver='lbfgs',alpha=1e-5,random_state=0)
dt=DecisionTreeClassifier(random_state=0)
rf=RandomForestClassifier(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
lr=LogisticRegression(random_state=0)
sc=svm.SVC(random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, test_size=0.2)
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
#
# sc.fit(x_train,y_train)
# y_pred4=sc.predict(x_test)
# print('SVM',accuracy_score(y_test,y_pred4))

gnb.fit(x_train,y_train)
y_pred5=gnb.predict(x_test)
print('Guassian naive bayes',accuracy_score(y_test,y_pred5))

# mnb.fit(x_train,y_train)
# y_pred5=mnb.predict(x_test)
# print('Multinomial naive bayes',accuracy_score(y_test,y_pred5))

from sklearn.naive_bayes import BernoulliNB
bn=BernoulliNB()
bn.fit(x_train,y_train)
pred=bn.predict(x_test)
print('Bernoullis Naive bayes',accuracy_score(y_test,pred))

mlp.fit(x_train,y_train)
pred1=mlp.predict(x_test)
print('Nueral networks',accuracy_score(y_test,pred1))

#Output
'''
Decision tree 0.8469230586736696
Random forest 0.8692448212744291
Gradient Boosting 0.8448243768283625
Logistic regression 0.7186735235773218
Guassian naive bayes 0.7698822730731819
Bernoullis Naive bayes 0.7187456087363563
Nueral networks 0.7186735235773218
'''
