# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("C:\\Users\\shs6g\\Desktop\\Data analysis Intern task\\input"))


# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import matplotlib.cm as cm

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

# associating the  data with a variable
telcom=pd.read_csv("C:\\Users\\shs6g\\Desktop\\Data analysis Intern task\\input\\Churn_data.csv")

# reading data
#print(telcom.head())

# printing meta data of data file
print(telcom.shape)

# look for null value
#print(pd.isnull(telcom).sum())

# check if dataset is balance or imbalance
print(telcom['Churn'].value_counts())

# print their data types
#print(telcom.dtypes)

# change the data type of  Total Charges from object to float
telcom['TotalCharges']= telcom['TotalCharges'].convert_objects(convert_numeric=True)
print(telcom['TotalCharges'].dtypes)

# check if TotalCharges column has null value
print(pd.isnull(telcom['TotalCharges']).sum())

# dropping the 11 null vlaues in TotalCharges
telcom.dropna(inplace=True)
print(telcom.shape)

# visualizing the proportions of Churn
churnvalue=telcom['Churn'].value_counts()
labels=telcom['Churn'].value_counts().index
rcParams['figure.figsize']=6,6
plt.pie(churnvalue,labels=labels,colors=['Green','Purple'],explode=(0.1,0),autopct='%1.1f%%',shadow=True)
plt.title('Churn Column Proportions')
plt.show()

# replacing the values in Churn column 'yes' with 1 and 'No' with '0'
telcom['Churn'].replace(to_replace='Yes', value=1, inplace=True)
telcom['Churn'].replace(to_replace='No', value=0, inplace=True)
print(telcom['Churn'].head())



f, axes=plt.subplots(nrows=2, ncols=2, figsize=(10,10))
plt.subplot(2,2,1)
gender=sns.countplot(x='gender', hue='Churn', data=telcom, palette='Pastel2')
plt.xlabel('gender')
plt.title('Churn by Gender')

plt.subplot(2,2,2)
seniorcitizen=sns.countplot(x="SeniorCitizen",hue="Churn",data=telcom,palette="Pastel2")
plt.xlabel("senior citizen")
plt.title("Churn by Senior Citizen")

plt.subplot(2,2,3)
partner=sns.countplot(x="Partner",hue="Churn",data=telcom,palette="Pastel2")
plt.xlabel("partner")
plt.title("Churn by Partner")

plt.subplot(2,2,4)
dependents=sns.countplot(x="Dependents",hue="Churn",data=telcom,palette="Pastel2")
plt.xlabel("dependents")
plt.title("Churn by Dependents")
plt.show()

# Finding corelation between variables
plt.figure(figsize=(20,16))
charges=telcom.iloc[:,1:20]
corr=charges.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=0.2, cmap='PuBuGn', annot= True)
plt.title("Variable Correlation")
plt.show()

tel_dummies = pd.get_dummies(telcom.iloc[:,1:21])
print(tel_dummies.head())

plt.figure(figsize=(15,8))
tel_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.title("Churn Correlation with other variables")
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()

covariables=["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(16,10))
for i, item in enumerate(covariables):
    plt.subplot(2,3,(i+1))
    ax=sns.countplot(x=item,hue="Churn",data=telcom,palette="Pastel2",order=["Yes","No","No internet service"])
    plt.xlabel(str(item))
    plt.title("Churn by "+str(item))
    i=i+1
plt.show()

sns.barplot(x="Contract",y="Churn", data=telcom, palette="Pastel1", order= ['Month-to-month', 'One year', 'Two year'])
plt.title("Churn by Contract type")
plt.show()

def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(telcom[telcom['Churn'] == 0 ][feature].dropna(), color= 'navy', label= 'Churn: No')
    ax1 = sns.kdeplot(telcom[telcom['Churn'] == 1 ][feature].dropna(), color= 'orange', label= 'Churn: Yes')
    plt.show()
kdeplot('tenure')
kdeplot('MonthlyCharges')
kdeplot('TotalCharges')
telcomvar=telcom.iloc[:,2:20]
telcomvar.drop(columns="PhoneService",axis=1, inplace=True)
print(telcomvar.head())
scaler = StandardScaler(copy=False)
print(scaler.fit_transform(telcomvar[['tenure','MonthlyCharges','TotalCharges']]))
telcomvar[['tenure','MonthlyCharges','TotalCharges']]=scaler.transform(telcomvar[['tenure','MonthlyCharges','TotalCharges']])
# check outliers
plt.figure(figsize = (8,4))
numbox = sns.boxplot(data=telcomvar[['tenure','MonthlyCharges','TotalCharges']], palette="Set2")
plt.title("Check outliers of standardized tenure, MonthlyCharges and TotalCharges")
plt.show()


def uni(columnlabel):
    print(columnlabel, "--", telcomvar[columnlabel].unique())


telcomobject = telcomvar.select_dtypes(['object'])
for i in range(0, len(telcomobject.columns)):
    uni(telcomobject.columns[i])

telcomvar.replace(to_replace='No internet service', value='No', inplace=True)
telcomvar.replace(to_replace='No phone service', value='No', inplace=True)
for i in range(0,len(telcomobject.columns)):
    uni(telcomobject.columns[i])

def labelencode(columnlabel):
    telcomvar[columnlabel] = LabelEncoder().fit_transform(telcomvar[columnlabel])

for i in range(0,len(telcomobject.columns)):
    labelencode(telcomobject.columns[i])

for i in range(0,len(telcomobject.columns)):
    uni(telcomobject.columns[i])

X=telcomvar
y=telcom["Churn"].values

sss=StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=0)
sss.get_n_splits(X,y)

print(sss)

#Split train/test sets of X and y
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train,X_test=X.iloc[train_index], X.iloc[test_index]
    y_train,y_test=y[train_index], y[test_index]

#Let's see the number of sets in each class of training and testing datasets
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())

Classifiers=[["Random Forest",RandomForestClassifier()],
             ["LogisticRegression",LogisticRegression()]
]
Classify_result=[]
names=[]
prediction=[]
for name,classifier in Classifiers:
    classifier=classifier
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    recall=recall_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    class_eva=pd.DataFrame([recall,precision])
    Classify_result.append(class_eva)
    name=pd.Series(name)
    names.append(name)
    y_pred=pd.Series(y_pred)
    prediction.append(y_pred)
names=pd.DataFrame(names)
names=names[0].tolist()
result=pd.concat(Classify_result,axis=1)
result.columns=names
result.index=["recall","precision"]
print(result)


prediction=pd.DataFrame(prediction)
y_pred_rf=np.array(prediction.iloc[0,:])
y_pred_lr=np.array(prediction.iloc[1,:])


predictions = [y_pred_rf, y_pred_lr]

import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True value')
    plt.xlabel('Predicted value')


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
for i, item in enumerate(predictions):
    plt.subplot(2,5,(i+1))
    cnf_matrix = confusion_matrix(y_test,item)
    class_names = ["Remain","Churn"]
    title_label=["RF","LR"]
    plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='CM of '+str(title_label[i]))
    i=i+1
plt.show()

#set the parameters by cross-validation for LogisticRegression
param_grid={'C':[0.001,0.01,0.1,1,10,100,1000],
            'solver':['warn','newton-cg', 'lbfgs', 'sag', 'saga'],
            'max_iter':[10,50,100,150,200,300],
            'class_weight':[None,'balanced'],
            'multi_class':['warn','ovr','auto']
           }
grid_LogReg=GridSearchCV(LogisticRegression(penalty='l2'),param_grid,cv=5,scoring=make_scorer(recall_score))
grid_LogReg=grid_LogReg.fit(X_train,y_train)

print("Optimized LogisticRegression:", grid_LogReg.best_estimator_)