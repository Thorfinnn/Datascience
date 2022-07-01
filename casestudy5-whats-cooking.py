import pandas as pd
df=pd.read_json('cousine_train.json')
# print(df['cuisine'].unique())
df_c=['greek' ,'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian',
 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']
x=df['ingredients']
y=df['cuisine'].apply(df_c.index)
df['all_ingredients']=df['ingredients'].map(';'.join)
print(df)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv=CountVectorizer()
x=cv.fit_transform(df['all_ingredients'])

print(x)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print('Random forest',accuracy_score(y_test,y_pred1))

gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print('Gradient boosting',accuracy_score(y_test,y_pred2))
#
# Random forest 0.7586423632935261
# Gradient boosting 0.7565053425518542
