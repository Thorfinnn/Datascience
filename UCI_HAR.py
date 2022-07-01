import pandas as pd
#reading feature names
features=list()
with open('UCI_HAR_features.txt') as feature:
    features=[line.split()[1] for line in feature.readlines()]
#reading training and train
x_train=pd.read_csv('X_train.txt',delim_whitespace=True,header=None)
x_train.columns=features
x_train['subject']=pd.read_csv('subject_train.txt',header=None,squeeze=True)
y_train=pd.read_csv('y_train.txt',names=['Activity'],squeeze=True)
y_train_labels=y_train.map({
    1:'Walking',
    2:'Walking_Upstairs',
    3:'Walking_Downstairs',
    4:'Sitting',
    5:'Standing',
    6:'Laying'
})
df=x_train
df['Activity']=y_train
df['ActivityName']=y_train_labels

#changing column names
columns=df.columns
columns=columns.str.replace('[()]','')
columns=columns.str.replace('[-]','-')
columns=columns.str.replace('[,]','')
df.columns=columns
#stationary and dynamic movements are different
#No Nan or empty values to fill
#No need for scaling as data in range (-1,1)

x=df.drop(['subject','Activity','ActivityName'],axis=1)
y=df.ActivityName

#oversampling
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
ros.fit(x,y)
x,y=ros.fit_resample(x,y)



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# bestfeatures = SelectKBest(score_func=chi2,k='all')
# bestfeatures.fit(x,y)
# dfscore=pd.DataFrame(bestfeatures.scores_)
# dfcolumns=pd.DataFrame(x.columns)
# featuresScores=pd.concat([dfcolumns,dfscore],axis=1)
# featuresScores.columns=['Features','Score']
# print(featuresScores)


# for col in df.columns.values:
#     sns.boxplot(df[col])
#     plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
# df['SibSp']=pd.cut(df['SibSp'],2,labels=[0,1])

mlp=MLPClassifier(solver='lbfgs',alpha=1e-5,random_state=0)
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


#OUTPUT
'''
Decision tree 0.9206429780033841
Random forest 0.9732656514382403
Gradient Boosting 0.9472081218274112
Logistic regression 0.9766497461928934
SVM 0.9590524534686972
Guassian naive bayes 0.6847715736040609
Bernoullis Naive bayes 0.8582064297800338
Nueral networks 0.9746192893401016
'''
