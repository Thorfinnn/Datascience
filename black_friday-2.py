import pandas as pd
import math
df=pd.read_csv('black_friday_train.csv')

# for col in df.columns.values:
#     df[col].fillna(0,inplace=True)
print(df.info())
df['Product_Category_2']=df['Product_Category_2'].fillna(0)
df['Product_Category_3']=df['Product_Category_3'].fillna(0)

print(df['Purchase'].mean())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# df=df.drop('Product_ID',axis=1 )
# df=df.drop('User_ID',axis=1 )
df['User_ID']=df['User_ID']-1e6
df['Product_ID']=df['Product_ID'].str.replace('P00','')\
                 # -142
df['Age']=df['Age'].map({
    '0-17':0,
    '18-25':1,
    '26-35':2,
    '36-45':3,
    '46-50':4,
    '51-55':5,
    '55+':6
})
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].map({
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4+':4
})

# df=df[~(df['Product_Category_3']=='3')]
# df=df[~(df['Product_Category_3']=='4')]
# df=df[~(df['Product_Category_3']=='10')]
# df=df[~(df['Product_Category_3']=='11')]
# df=df[~(df['Product_Category_3']=='18')]



#Labeling Categorical or range values
for col in ['Gender','City_Category']:
    df[col]=le.fit_transform(df[col])
df=df.drop('Product_ID',axis=1 )
df=df.drop('User_ID',axis=1 )

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
num_cols=['Age','Occupation','Stay_In_Current_City_Years','Product_Category_1','Product_Category_2','Product_Category_3']
for col in num_cols:
    df[col]=scaler.fit_transform(df[col].values.reshape(-1,1))
# import seaborn as sns
# from matplotlib import pyplot as plt
# for col in df.columns.values:
#     sns.boxplot(df[col])
#     plt.show()
df=df[~(df['Age']>1.8)]
df=df[~(df['Age']<-1.8)]
df=df[~(df['Product_Category_1']>=3)]
df=df[~(df['Purchase']>2.142e4)]
# df=df.drop('Gender',axis=1)
df['Purchase']=df['Purchase']-9263
x=df.drop('Purchase',axis=1)
y=df['Purchase']
# for col in num_cols:
#     q1=df[col].quantile(0.25)
#     q3=df[col].quantile(0.75)
#     r=q3-q1
#     upper=q3+1.5*r
#     lower=q1-1.5*r
#     o1=df[df[col]<lower].values
#     o2=df[df[col]>upper].values
#     df[col].replace(o1,lower,inplace=True)
#     df[col].replace(o2,upper,inplace=True)
# print(x['Product_ID'].min())

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# bestfeatures = SelectKBest(score_func=chi2,k='all')
# bestfeatures.fit(x,y)
# dfscore=pd.DataFrame(bestfeatures.scores_)
# dfcolumns=pd.DataFrame(x.columns)
# featuresScores=pd.concat([dfcolumns,dfscore],axis=1)
# featuresScores.columns=['Features','Score']
# print(featuresScores)
#                      Features         Score
# 0                      Gender  5.861838e+03
# 1                         Age  1.683454e+04
# 2                  Occupation  9.884267e+04
# 3               City_Category  1.238436e+04
# 4  Stay_In_Current_City_Years  1.613194e+04
# 5              Marital_Status  1.083794e+04
# 6          Product_Category_1  1.106865e+06
# 7          Product_Category_2  2.307722e+05
# 8          Product_Category_3  1.188604e+06

df=df.drop(['Gender','Marital_Status'],axis=1,inplace=True)
# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
#
# et = ExtraTreesClassifier()
# et.fit(x,y)
# print(et.feature_importances_)
#
# feat_imp=pd.Series(et.feature_importances_,index=x.columns)
# feat_imp.nlargest(9).plot(kind='bar')
# plt.show()
# from sklearn.decomposition import PCA
# pca=PCA(n_components=5)
# pca.fit(x)
# x=pca.transform(x)
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import L
from sklearn.model_selection import train_test_split
lr=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import mean_squared_error
print('Linear Regression',math.sqrt(mean_squared_error(y_test,y_pred)))
# print(y_test,y_pred)
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
print('Decission Tree',math.sqrt(mean_squared_error(y_test,y_pred)))
# print(y_test,y_pred)
rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print('Random Forest',math.sqrt(mean_squared_error(y_test,y_pred1)))
# print(y_test,y_pred1)
gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print('Gradient Boosting',math.sqrt(mean_squared_error(y_test,y_pred2)))
# print(y_test,y_pred2)
# sv.fit(x_train,y_train)
# y_pred4=sv.predict(x_test)
# print(mean_squared_error(y_test,y_pred4))

# Linear Regression 4527.190986108847
# Decission Tree 3318.2049783164075
# Random Forest 3058.2449600760656
# Gradient Boosting 2996.669669365222

