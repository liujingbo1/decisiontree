#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
train = pd.read_csv("F:\\kaggle_credcard\\Titanic\\train.csv")
test = pd.read_csv("F:\\kaggle_credcard\\Titanic\\test.csv")
print("Dimension of train data {}".format(train.shape))
print("Dimension of test data {}".format(test.shape))

#%%
#train.tail()

#查看数据的分布特点
#a =train.describe()
#print(a)

#Passengerld表示乘客的id号，Survived生存的标志0-遇难，1-幸存，Pclass船舱的等级，3等表示最普通的
#Name表示乘客姓名，Sex性别，Age年龄，Sibsp兄弟姐妹，Parch父母小孩，Ticket票的编号，Fare费用，
#Cabin仓号，Embarked上船的地方

#%%
#观察测试集与训练集数据的特征是否相似
plt.rc('font',size =13),
fig = plt.figure(figsize=(18,8))
alpha =0.6

ax1 = plt.subplot2grid((2,3),(0,0))
train.Age.value_counts().plot(kind = 'kde',color ='#FA2379',label='train',alpha=alpha)
test.Age.value_counts().plot(kind = 'kde',label = 'test',alpha = alpha)
ax1.set_xlabel('Age')
ax1.set_title("What's the distribution of age?")
plt.legend(loc = 'best')


ax2 = plt.subplot2grid((2,3),(0,1))
train.Pclass.value_counts().plot(kind = 'barh',color = '#FA2379',label='train',alpha = alpha)
test.Pclass.value_counts().plot(kind = 'barh',label =  'test',alpha = alpha)
ax2.set_ylabel('Pclass')
ax2.set_xlabel('Frequency')
ax2.set_title("What's the distribution of Pclass")
plt.legend(loc= 'best')

ax3 = plt.subplot2grid((2,3),(0,2))
train.Sex.value_counts().plot(kind = 'barh',color = '#FA2379',label = 'train',alpha = alpha)
test.Sex.value_counts().plot(kind='barh',label = 'test',alpha = alpha)
ax3.set_ylabel('Sex')
ax3.set_xlabel('Frequency')
ax3.set_title("What's the distribution of Sex")
plt.legend(loc = 'best')

ax4 = plt.subplot2grid((2,3),(1,0),colspan =2)
train.Fare.value_counts().plot(kind = 'kde',color ='#FA2379',label ='train',alpha = alpha)
test.Fare.value_counts().plot(kind = 'kde',label = 'test',alpha = alpha)
ax4.set_xlabel('Fare')
ax4.set_title("What's the distribution of Fare?")
plt.legend(loc = 'best')


ax5 = plt.subplot2grid((2,3),(1,2) )
train.Embarked.value_counts().plot(kind = 'barh',color ='#FA2379',label = 'train',alpha = alpha)
test.Embarked.value_counts().plot(kind = 'barh',label = 'test',alpha =  alpha)
ax5.set_ylabel('Embarked')
ax5.set_xlabel('Frequency')
ax5.set_title("What's the distribution of Embarked?")
plt.legend(loc ='best')
plt.tight_layout()

#%%%
print(train.Survived.value_counts())
#0没有幸存，1幸存
#%%
num_null = train.isnull().sum()

#%%%
#10岁以下的孩子中幸存的比例比遇难的比例高
fig = plt.figure(figsize = (15,6))
train[train.Survived ==0].Age.value_counts().plot(
        kind ='density',color ='#FA2379',label ='Not Survived' ,alpha = alpha)
train[train.Survived ==1].Age.value_counts().plot(
        kind='density',label='Survived',alpha=alpha)
plt.xlabel('Age')
plt.title("What's the distribution of Age?")
plt.legend(loc = 'best')
plt.grid()

#%%
#女性幸存的比例比男性高出50%
df_male = train[train.Sex =='male'].Survived.value_counts().sort_index()
df_female = train[train.Sex == 'female'].Survived.value_counts().sort_index()
fig  = plt.figure(figsize = (18,6))
ax1= plt.subplot2grid((1,2),(0,0))
df_female.plot(kind = 'barh',color='#FA2379',label = 'Female',alpha = alpha)
df_male.plot(kind = 'barh',label='Male',alpha = alpha)
ax1.set_xlabel('Frequency')
ax1.set_yticklabels(['Died','Survived'])
ax1.set_title("Who will survived with respect to Sex?")
plt.legend(loc = 'best')
plt.grid()


ax2 = plt.subplot2grid((1,2),(0,1))
(df_female/train[train.Sex =='female'].shape[0]).plot(kind ='barh',color ='#FA2379',
label ='Female',alpha= alpha)
(df_male/train[train.Sex == 'male'].shape[0]).plot(kind = 'barh',label = 'male',alpha = alpha)
ax2.set_xlabel('Rate')
ax2.set_yticklabels(['Died','Survived'])
ax2.set_title("What's the survived rate with respect to Sex?")
plt.legend(loc = 'best')
plt.grid()


#%%
#观察座舱对男女获救的影响
df_male = train[train.Sex =='male']
df_female = train[train.Sex == 'female']
fig = plt.figure(figsize=(18,6))


ax1 = plt.subplot2grid((1,4),(0,0))
df_female[df_female.Pclass<3].Survived.value_counts().sort_index().plot(kind = 'bar',
         color = '#FA2379',alpha=alpha)
ax1.set_ylabel('Frequency')
ax1.set_ylim(0,350)
ax1.set_xticklabels(['Died','Survived'])
ax1.set_title("How will high-class female survived?",y=1.05)
plt.grid()


ax2 = plt.subplot2grid((1,4),(0,1))
df_female[df_female.Pclass ==3].Survived.value_counts().sort_index().plot(kind = 'bar',
         color = '#23FA79',alpha= alpha)
ax2.set_ylabel('Frequency')
ax2.set_ylim((0,350))
ax2.set_xticklabels(['Died','Survived'])
ax2.set_title("How will low-class female survived?",y=1.05)
plt.grid()

ax3 = plt.subplot2grid((1,4),(0,2))
df_male[df_male.Pclass <3].Survived.value_counts().sort_index().plot(kind ='bar',
       color = '#00FA23',alpha =alpha)
ax3.set_ylabel('Frequency')
ax3.set_ylim((0,350))
ax3.set_xticklabels(['Died','Survived'])
ax3.set_title("How will high-class male survived?",y=1.05)
plt.grid()

ax4 = plt.subplot2grid((1,4),(0,3))
df_male[df_male.Pclass==3].Survived.value_counts().sort_index().plot(kind = 'bar',
       color = '#2379FA',alpha = alpha)
ax4.set_ylabel('Frequency')
ax4.set_ylim((0,350))
ax4.set_xticklabels(['Died','Survived'])
ax4.set_title("How will low-class male survived?",y=1.05)
plt.grid()
plt.tight_layout()

#%%
#同一类票号可能是人员比较类似，比如同学、朋友或者家人
same_ticket1601 = train[train.Ticket == '1601']
same_ticketCA_2144 = train[train.Ticket =='CA 2144']



#%%
#数据清理
train_num_null = train.isnull().sum()
test_num_null = test.isnull().sum()
#%%
#假设Embarked上船的地方与Fare和pclass有关，填充Embarked的值
fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot(111)
ax = train.boxplot(column = 'Fare',by=['Embarked','Pclass'],ax=ax)
plt.axhline(y=80,color='green')
ax.set_title('',y=1.1)
null_Embarked = train[train.Embarked.isnull()][['Fare','Pclass','Embarked']]

#%%
train.at[train.Embarked.isnull(),'Embarked'] = 'C'

#%%
#通过结合Embarked和Pclass可以画出Fare的柱状图，采用频次最多的数作为填充值
fig = plt.figure(figsize =(8,5))
ax = fig.add_subplot(111)
test[(test.Pclass ==3) &(test.Embarked=='S')].Fare.hist(bins=100,ax=ax)
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title("Histogram of Faec ,Pclass 3 and Embarked S")

#%%
print("null of Fare all")
print(test[test.Fare.isnull()][['Pclass','Fare','Embarked']])
print("The top5 most common of Fare")
print(test[(test.Pclass==3) & (test.Embarked =='S')].Fare.value_counts().head())

#%%
test.at[test.Fare.isnull(),'Fare'] = 8.05

#%%
#将所有的cabin填充为U0
full = pd.concat([train,test],ignore_index = True)
full.at[full.Cabin.isnull(),'Cabin'] = 'U0'



#%%
#年龄先不处理，现在开始做特征工程
import re
names = full.Name.map(lambda x:len(re.split(' ' ,x)))
full.at[full.index,"Names"] = names
#del names

#%%
#create feature ,Title

title = full.Name.map(lambda x:re.compile(',(.*?)\.').findall(x)[0])
title[title =='Mme'] ='Mrs'
title[title.isin(['Ms','Mlle'])] = 'Miss'
title[title.isin(['Don','Jonkheer'])] = 'Sir'
title[title.isin(['Dona','Lady','the Countess'])] = 'Lady'
title[title.isin(['Capt','Col','Major','Dr','Officer','Rev'])] = 'Officer'
full.at[full.index,'Title'] = title
#del title
#%%
deck = full[~full.Cabin.isnull()].Cabin.map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
deck = pd.factorize(deck)[0]
full.at[full.index,'Deck'] = deck
#del deck
#%%
checker = re.compile("([0-9]+)")
def roomNum(x):
    nums = checker.search(x)
    if nums:
        return int(nums.group()) +1
    else:
        return 1
rooms = full.Cabin.map(roomNum)
full.at[full.index,'Room'] = rooms
full['Room'] = full.Room/full.Room.sum()
#del checker,roomNum
#%%
full['Group_num'] = full.Parch + full.SibSp +1
#%%
full['Group_size'] = pd.Series('M',index = full.index)
full.at[full.Group_num>4,'Group_size'] = 'L'
full.at[full.Group_num ==1,'Group_size'] = 'S'
#%%
#Normalized the fare
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
full['NorFare'] = pd.Series(scaler.fit_transform(full.Fare.values.reshape(-1,1)).reshape(-1),index = full.index)
#%%

def setValue(col):
    train.at[train.index,col]= full[:891][col].values
    test.at[test.index,col] = full[891:][col].values
for col in['Deck','Room','Group_size','Group_num','Names','Title']:
    setValue(col)
#%%
#Predict Age
full.drop(labels = ['PassengerId','Name','Cabin','Survived','Ticket','Fare'],axis =1,inplace = True)
full = pd.get_dummies(full,columns = ['Embarked','Sex','Title','Group_size'])
#%%
from sklearn.model_selection import train_test_split
X = full[~full.Age.isnull()].drop('Age',axis =1)
#%%
y = full[~full.Age.isnull()].Age
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42)
#%%
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
def get_model(estimator,parameters,X_train,y_train,scoring):
    model = GridSearchCV(estimator,param_grid = parameters,scoring = scoring)
    model.fit(X_train,y_train)
    return model.best_estimator_
import xgboost as xgb
#%%
XGB = xgb.XGBRegressor(max_depth =4,seed =42)
scoring = make_scorer(mean_absolute_error,greater_is_better= False)
parameters = {'reg_alpha' :np.linspace(0.1,1.0,5),'reg_lambda':np.linspace(1.0,3.0,5)}
reg_xgb = get_model(XGB,parameters,X_train,y_train,scoring)
print(reg_xgb)
#%%
print("Mean absolute error of test data: {}".format(mean_absolute_error(y_test,reg_xgb.predict(X_test))))

#%%
fig = plt.figure(figsize = (15,6))
alpha = 0.5
full.Age.value_counts().plot(kind = 'density',color = '#FA2379',label = 'Before',alpha= alpha)
pred = reg_xgb.predict(full[full.Age.isnull()].drop('Age',axis =1))
full.at[full.Age.isnull(),"Age"] = pred

full.Age.value_counts().plot(kind = 'density',label = 'After',alpha = alpha)
plt.xlabel('Age')
plt.title("What's the distribution of Age after predicting?")
plt.legend(loc = 'best')
plt.grid()
#%%

full['NorAge'] = pd.Series(scaler.fit_transform(full.Age.values.reshape(-1,1)).reshape(-1),index = full.index)
full['NorNames'] = pd.Series(scaler.fit_transform(full.Names.values.reshape(-1,1)).reshape(-1),index=full.index)
full['Group_num'] = pd.Series(scaler.fit_transform(full.Group_num.values.reshape(-1,1)).reshape(-1),index = full.index)
#%%
for col in ['NorAge','NorFare','NorNames','Group_num']:
    setValue(col)

#%%
train.Sex = np.where(train.Sex =='female',0,1)
test.Sex = np.where(test.Sex == 'female',0,1)

#%%
train.drop(labels = ['PassengerId','Name','Names','Cabin','Ticket','Age','Fare'],axis =1,inplace = True)
test.drop(labels = ['Name','Names','Cabin','Ticket','Age','Fare'],axis=1,inplace = True)
#%%

train = pd.get_dummies(train,columns =['Embarked','Pclass','Title','Group_size'])
test = pd.get_dummies(test,columns = ['Embarked','Pclass','Title','Group_size'])
test['Title_Sir'] = pd.Series(0,index = test.index)
#%%#
#Build Model
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator,title,X,y,ylim = None,cv = None,n_jobs =1,
    train_sizes = np.linspace(0.1,1.0,5),scoring = 'accuracy'):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes,train_scores,test_scores = learning_curve(
            estimator,X,y,cv=cv,scoring = scoring,n_jobs = n_jobs,train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis =1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                     train_scores_mean + train_scores_std,alpha =0.1,
                     color = "r")
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,alpha =0.1,
                     color = "g")
    plt.plot(train_sizes,train_scores_mean,'o-',color = "r",label = "Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color = "g",label = "Cross-validation score")
    plt.legend(loc = "best")
    return plt


#%%

from sklearn.metrics import roc_curve,auc
def plot_roc_curve(estimator,X,y,title):
    fpr,tpr,_ = roc_curve(y,estimator.predict_prob(X)[:,1])
    roc_auc = auc(fpr,tpr)
    print('ROC AUC: %0.2f'%roc_auc)
    plt.figure(figsize=(10,6))
    plt.plot(fpr,tpr,label = "ROC curve(area = %0.2f"%roc_auc)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,10])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - {}".format(title))
    plt.legend(loc = 'lower right')
    plt.show()
#%%
X = train.drop(['Survived'],axis =1)
y = train.Survived
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state =42)
from sklearn.metrics import accuracy_score
scoring = make_scorer(accuracy_score,greater_is_better=True)
#%%  
#KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(weights = 'uniform') 
parameters = {'n_neighbors':[3,4,5],'p':[1,2]}
clf_knn = get_model(KNN,parameters,X_train,y_train,scoring) 
#%%
print(accuracy_score(y_test,clf_knn.predict(X_test)))
print(clf_knn)
plot_learning_curve(clf_knn,'KNN',X,y,cv=4)    
#%%
#RamdomForestClassifer
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 42,criterion = 'entropy',
                             min_samples_split=5,oob_score = True)
parameters = {'n_estimators' :[500],'min_samples_leaf':[12]}
clf_rfc1 = get_model(rfc,parameters,X_train,y_train,scoring)
print(accuracy_score(y_test,clf_rfc1.predict(X_test)))
print(clf_rfc1)
plot_learning_curve(clf_rfc1,'Random Forest',X,y,cv=4)   
    
 #%%   
plt.figure(figsize = (10,6))
plt.barh(np.arrange(X_train.columns.shape[0]),clf_rfc1.features_,0.5)  
plt.yticks(np.arrange(X_train.columns.shape[0]),X_train.columns)
plt.grid()
plt.xticks(np.arange(0,0.2,0.02))
    
#%%
cols = X_train.columns[clf_rfc1.feature_importances_ >=0.016]
rfc = RandomForestClassifier(random_state =42,criterion='entropy',
                             min_samples_split =5,oob_score = True)
parameters = {'n_estimators':[500],'min_samples_leaf':[12]}
clf_rfc2 = get_model(rfc,parameters,X_train[cols],y_train,scoring)
#%%

print(clf_rfc2)
print(accuracy_score(y_test,clf_rfc2.predict(X_test[cols])))
plot_learning_curve(clf_rfc2,'Ramdom Forest',X[cols],y,cv=4)
#%%
#逻辑回归
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state = 42,penalty = 'l1',solver = 'liblinear')
parameters = {'C':[0.5]}
clf_lg1 = get_model(lg,parameters,X_train,y_train,scoring)
print(clf_lg1)
print(accuracy_score(y_test,clf_lg1.predict(X_test)))
plot_learning_curve(clf_lg1,'Logistic Regression',X,y,cv=4)

#%%
#SVC
from sklearn.svm import SVC
svc = SVC(random_state =42,kernel = 'poly',probability = True)
parameters = {'C' :[35],'gamma' :[0.0055],'coef0':[0.1],'degree':[2]}
clf_svc = get_model(svc,parameters,X_train,y_train,scoring)
print(clf_svc)
print(accuracy_score(y_test,clf_svc.predict(X_test)))
plot_learning_curve(clf_svc,'SVC',X,y,cv=4)
#%%
#XGBoost
import xgboost as XGB
xgb = XGB.XGBClassifier(seed =42,max_depth =3,objective ='binary:logistic',
                        n_estimators =400)
parameters = {'learning_rate':[0.1],'reg_alpha':[3.0],'reg_lambda':[4.0]}
clf_xgb1 = get_model(xgb,parameters,X_train,y_train,scoring)
print(accuracy_score(y_test,clf_xgb1.predict(X_test)))
print(clf_xgb1)
plot_learning_curve(clf_xgb1,'XGB',X,y,cv=4)
#%%
#Ensemble
from sklearn.ensemble import VotingClassifier
clf_vc = VotingClassifier(estimators = [('xgb1',clf_xgb1),('lg1',clf_lg1),('svc',clf_svc),
        ('rfc1',clf_rfc1),('rfc2',clf_rfc2),('knn',clf_knn)],voting = 'hard',weights =[4,1,1,1,1,2])
clf_vc = clf_vc.fit(X_train,y_train)
print(accuracy_score(y_test,clf_vc.predict(X_test)))
print(clf_vc)
plot_learning_curve(clf_vc,'Ensemble',X,y,cv=4)
#%%
