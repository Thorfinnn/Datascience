import pandas as pd
df=pd.read_csv('titanic.csv')
df=df.drop('PassengerId',axis=1)
df=df.drop('Cabin',axis=1)
df=df.drop('Name',axis=1)
df=df.drop('Ticket',axis=1)
df=df.drop('Embarked',axis=1)
df=df.drop('Fare',axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Sex'])
df['Sex']=le.transform(df['Sex'])
df['Age'].fillna(df['Age'].median(),inplace=True)
print(df.describe())

x=df.drop('Survived',axis=1)
y=df['Survived']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2,k='all')
bestfeatures.fit(x,y)
dfscore=pd.DataFrame(bestfeatures.scores_)
dfcolumns=pd.DataFrame(x.columns)
featuresScores=pd.concat([dfcolumns,dfscore],axis=1)
featuresScores.columns=['Features','Score']
print(featuresScores)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


