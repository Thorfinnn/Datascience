import pandas as pd
store=pd.read_csv('walmart_stores.csv')
train=pd.read_csv('walmart_train.csv')
features=pd.read_csv('walmart_features.csv')
df=train.merge(store,how='inner',on='Store')\
    .merge(features,how='inner',on=['Store','Date','IsHoliday'])\
    .reset_index(drop=True).sort_values(['Store','Dept','Date'])
df['Date']=pd.to_datetime(df['Date'])
df['Day']=df['Date'].dt.isocalendar().day
df['Week']=df['Date'].dt.isocalendar().week
df['Month']=df['Date'].dt.month
df['Year']=df['Date'].dt.year
from sklearn.preprocessing import LabelEncoder

df.fillna(0,inplace=True)
#removing negative weekly sales
df=df[df['Weekly_Sales']>=0]
df.reset_index(inplace=True,drop=True)
#turning negative markdown to zero
df.loc[df['MarkDown2']<0,'MarkDown2']=0
df.loc[df['MarkDown3']<0,'MarkDown3']=0
#fuel price not effect weekly sales significantly
df=df.drop('Fuel_Price',axis=1)
#CPI does not effect weekly fuel price significantly
df=df.drop('CPI',axis=1)
df=df.drop('Date',axis=1)

#fixing isholiday with values 0 and 1
df['IsHoliday'].apply(lambda x:1 if x else 0)
# print(df.columns.values)

#labeling store types
le=LabelEncoder()
df['Type']=le.fit_transform(df['Type'])

x=df.drop('Weekly_Sales',axis=1)
y=df['Weekly_Sales']
#relavant Features
rfeatures=['Dept','Size','Store','Week','Type','IsHoliday','Month','Year']
for col in x.columns.values:
    if col not in rfeatures:
        x=x.drop(col,axis=1)
print(x.describe())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
lr=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import mean_squared_error
print('Linear regression',mean_squared_error(y_test,y_pred))
# print(y_test,y_pred)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
rf=RandomForestRegressor(n_estimators=58, max_depth=27, min_samples_leaf=1,min_samples_split=3,n_jobs=-1)
gb=GradientBoostingRegressor()
sv=svm.SVR()
dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print('Decision Tree',mean_squared_error(y_test,y_pred))
# print(y_test,y_pred)
rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print('Random Forest',mean_squared_error(y_test,y_pred1))
# print(y_test,y_pred1)
gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print('Gradient Boosting',mean_squared_error(y_test,y_pred2))
# print(y_test,y_pred2)
# sv.fit(x_train,y_train)
# y_pred4=sv.predict(x_test)
# print(mean_squared_error(y_test,y_pred4))

#OUTPUT
# Linear regression 474045986.5850748
# Decision Tree 16661542.29635858
# Random Forest 10700016.99513372
# Gradient Boosting 139539271.96147263
