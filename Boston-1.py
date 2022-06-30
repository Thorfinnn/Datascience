import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df=pd.read_csv('housing.csv',header=None,delimiter=r'\s+',names=column_names)
df['CHAS'].fillna(0,inplace=True)
# df[]
column_names.remove('CHAS')
for col in column_names:
    if df[col].count()!=506:
        df[col].fillna(df[col].max(),inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt

columns=[
# 'CRIM',
'ZN',
# 'RM',
'DIS',
# 'PTRATIO',
# 'B',
'LSTAT',
# 'MEDV'
]

for col in columns:
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    r=q3-q1
    upper=q3+1.5*r
    lower=q1-1.5*r
    o1=df[df[col]<lower].values
    o2=df[df[col]>upper].values
    df[col].replace(o1,lower,inplace=True)
    df[col].replace(o2,upper,inplace=True)
df=df[~(df['MEDV']>=50.0)]
df=df[~(df['CRIM']>=25.0)]
df=df[~(df['NOX']>=0.85)]
# df=df[~(df['RM']<=4.0)]
df=df[~(df['B']<300.0)]
print(df.describe())
# df['CHAS']=pd.cut(df['CHAS'],3,labels=[0,1,2])
# df['ZN']=pd.cut(df['ZN'],2,labels=[0,1])
df['B']=pd.cut(df['B'],2,labels=[0,1])
# df['TAX']=pd.cut(df['TAX'],4,labels=range(0,4))
# for col in df.columns.values:
#     sns.boxplot(df[col])
#     plt.show()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x=df.drop('MEDV',axis=1)
y=df['MEDV']
# print(x)
# print(y)
# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
#
# et = ExtraTreesClassifier()
# et.fit(x,y)
# print(et.feature_importances_)
#
# feat_imp=pd.Series(et.feature_importances_,index=x.columns)
# feat_imp.nlargest(4).plot(kind='bar')
# plt.show()

# print(x.describe())
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
# x=x.drop('AGE',axis=1)
x=x.drop('ZN',axis=1)
# x=x.drop('CHAS',axis=1)
# x=x.drop('RAD',axis=1)
# x=x.drop('LSTAT',axis=1)

print(x.columns.values)
lr=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
rf=RandomForestRegressor()
gb=GradientBoostingRegressor()
sv=svm.SVR()
dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(mean_squared_error(y_test,y_pred))

rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print(mean_squared_error(y_test,y_pred1))

gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print(mean_squared_error(y_test,y_pred2))

sv.fit(x_train,y_train)
y_pred4=sv.predict(x_test)
print(mean_squared_error(y_test,y_pred4))
#
# 11.945754994606157
# 8.750714285714286
# 7.245825607142856
# 6.737593926811454
# 47.9731716301474

