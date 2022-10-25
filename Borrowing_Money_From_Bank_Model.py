# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:12:37 2022

@author: ANANDAN RAJU
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(18,8)})
sns.set_palette('rainbow')

money=pd.read_csv('train.csv')

cred_dict={'Poor':1,'Standard':2,'Good':3}
for key,val in cred_dict.items():
    money['Credit_Score']=money['Credit_Score'].replace(key,val)
    
plt.title('Month Vs Credit Score',fontsize=15)
sns.barplot(data=money,x='Month',y='Credit_Score')
plt.show()

plt.title('Month',fontsize=15)
sns.distplot(money['Month'])
plt.show()

sns.histplot(money['Age'],kde=True)
plt.show()

age_round=[i//10 for i in money['Age']]
money['Age_Range']=age_round

plt.title('Age_Range Vs Credit Score')
sns.barplot(x='Age_Range',y='Credit_Score',data=money)
plt.show()

plt.title('Age_Range')
sns.countplot(x='Age_Range',hue='Credit_Score',data=money)
plt.show()

plt.title("Occupation Vs Credit Score", fontsize=15)
sns.barplot(x="Credit_Score", y="Occupation", data=money)
plt.show()

sns.barplot(y="Annual_Income",x='Credit_Score',data=money)
plt.show()

plt.title("Number of Bank Accounts", fontsize=15)
sns.countplot(y="Num_Bank_Accounts", data=money,color='blue')
plt.show()

plt.title('Number of Bank Accounts Vs Credit Score',fontsize=15)
sns.barplot(x="Num_Bank_Accounts", y="Credit_Score", data=money)
plt.show()

plt.title("Number of Bank Accounts Vs Credit Score", fontsize=15)
sns.lineplot(x="Num_Bank_Accounts", y="Credit_Score", data=money)
plt.show()

plt.title("Num Credit Card", fontsize=15)
sns.countplot(y="Num_Credit_Card", data=money,color='red')
plt.show()

plt.title("Num Credit Card Vs Credit Score", fontsize=15)
sns.barplot(x="Num_Credit_Card", y="Credit_Score", data=money)
plt.show()

plt.title("Num Credit Card Vs Credit Score", fontsize=15)
sns.lineplot(x="Num_Credit_Card", y="Credit_Score", data=money)
plt.show()

plt.title("Interest Rate", fontsize=20)
sns.countplot(y="Interest_Rate", data=money,color='orange')
plt.show()

plt.title("Interest Rate Vs Credit Score", fontsize=15)
sns.barplot(x="Interest_Rate", y="Credit_Score", data=money)
plt.show()

plt.title("Interest Rate Vs Credit Score", fontsize=15)
sns.lineplot(x="Interest_Rate", y="Credit_Score", data=money)
plt.show()

plt.title("Num of Loan", fontsize=20)
sns.countplot(y="Num_of_Loan", data=money,color='green')
plt.show()

plt.title('Number of Loan Vs Credit Score',fontsize=15)
sns.barplot(x="Num_of_Loan", y="Credit_Score", data=money)
plt.show()

plt.title("Number of Loan Vs Credit Score", fontsize=15)
sns.lineplot(x="Num_of_Loan", y="Credit_Score", data=money)
plt.show()

plt.figure(figsize=(20,14))
plt.title("Number of Bank Accounts", fontsize=15)
sns.countplot(y="Delay_from_due_date", data=money,color='brown')
plt.show()

plt.title('Delay from due date Vs Credit Score',fontsize=15)
sns.barplot(x="Delay_from_due_date", y="Credit_Score", data=money)
plt.show()

plt.title("Delay from due date Vs Credit Score", fontsize=15)
sns.lineplot(x="Delay_from_due_date", y="Credit_Score", data=money)
plt.show()

plt.title("Num of Delayed Payment", fontsize=20)
sns.countplot(y="Num_of_Delayed_Payment", data=money,color='grey')
plt.show()

plt.title("Num of Delayed Payment Vs Credit Score", fontsize=20)
sns.barplot(x="Num_of_Delayed_Payment", y="Credit_Score", data=money)
plt.show()

plt.title("Num of Delayed Payment Vs Credit Score", fontsize=20)
sns.lineplot(x="Num_of_Delayed_Payment", y="Credit_Score", data=money)
plt.show()

plt.title("Number of Credit Inquiries", fontsize=20)
sns.countplot(x="Num_Credit_Inquiries", data=money,color='violet')
plt.show()

plt.title("Number of Credit Inquiries Vs Credit Score", fontsize=20)
sns.barplot(x="Num_Credit_Inquiries", y="Credit_Score", data=money)
plt.show()

plt.title("Number of Credit Inquiries Vs Credit Score", fontsize=20)
sns.lineplot(x="Num_Credit_Inquiries", y="Credit_Score", data=money)
plt.show()

plt.title("Credit Mix", fontsize=15)
sns.barplot(x="Credit_Mix", y="Credit_Score", data=money)
plt.show()

plt.title("Payment of Min Amount", fontsize=15)
sns.barplot(x="Payment_of_Min_Amount", y="Credit_Score", data=money)
plt.show()

plt.title("Payment Behaviour", fontsize=20)
sns.countplot(y="Payment_Behaviour", data=money)
plt.show()

plt.title("Payment of Behaviour Vs Credit Score", fontsize=15)
sns.barplot(y="Payment_Behaviour", x="Credit_Score", data=money)
plt.show()

sns.histplot(x="Monthly_Balance", hue="Credit_Score",data=money,kde = True,multiple='stack')
plt.title("Monthly Balance", fontsize=20)
plt.show()

sns.histplot(x="Monthly_Inhand_Salary", hue="Credit_Score",data=money,multiple='stack',kde = True)
plt.title("Monthly Inhand Salary", fontsize=15)
plt.show()

# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

x=money.drop(['Payment_Behaviour','Payment_of_Min_Amount','Type_of_Loan','Name', 'Credit_Mix','Occupation',"Credit_Score"],axis=1)
y=money.iloc[:,-2]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)

test=pd.read_csv('test.csv')

t_age_range=[i//10 for i in test['Age']]
test['Age_Range']=t_age_range

t_x_test=test.drop(['Payment_Behaviour','Payment_of_Min_Amount','Type_of_Loan','Name', 'Credit_Mix','Occupation'], axis=1)

pred=rfc.predict(t_x_test)

rfc_score=rfc.score(x_test,y_test)*100

clf = [DecisionTreeClassifier(),ExtraTreeClassifier(),BaggingClassifier(),ExtraTreesClassifier()]

def model_fitting():
    for i in range(len(clf)):
        fit=clf[i].fit(x_train,y_train)
        print(fit)
model_fitting()

model=['Decision Tree Classifier    ','Extra Tree Classifier       ',
       'Bagging Classifier          ','Extra Trees Classifier      ']
accuracy=['KNC_Acc','DTC_Acc','GNB-Acc','ETC_Acc','GBC_Acc','BC_Acc','ABC_Acc','ETsC_Acc']

def model_accuracy():
    print("Accuracy Score of Each Model\n")
    for i in range(len(clf)):
        accuracy[i]=clf[i].score(x,y)*100
        print(model[i],':',round(accuracy[i],3))
model_accuracy()

test["Pred_Credit_Score"] = pred

pred_dict = {1:"Poor", 2:"Standard", 3:"Good"}

for key, val in pred_dict.items():
    test["Pred_Credit_Score"] = test["Pred_Credit_Score"].replace(key,val)
    
good_credit=test[test['Pred_Credit_Score']=='Good']
standard_credit=test[test['Pred_Credit_Score']=='Standard']
poor_credit=test[test['Pred_Credit_Score']=='Poor']

good_job=pd.DataFrame({'Job':test[test['Pred_Credit_Score']=='Good']['Occupation'],'credit_score':good_credit['Pred_Credit_Score']})

plt.title("Occupations by Prediction", fontsize=20)
sns.countplot(data=test, y="Occupation", hue="Pred_Credit_Score")
plt.show()

plt.title('Good Credit Score Occupations',fontsize=15)
sns.countplot(data=good_job,y='Job')
plt.show()

standard_job = pd.DataFrame({"Job": test[test["Pred_Credit_Score"] == "Standard"]["Occupation"], "Credit_Score": standard_credit["Pred_Credit_Score"]})

plt.title('Standard Credit Score Occupations',fontsize=15)
sns.countplot(data=standard_job,y='Job')
plt.show()

poorjob = pd.DataFrame({"Job": test[test["Pred_Credit_Score"] == "Poor"]["Occupation"], "Credit_Score": poor_credit["Pred_Credit_Score"]})

plt.title("Poor Occupations", fontsize=20)
sns.countplot(data=poorjob, y="Job")
plt.show()

goodbehav = pd.DataFrame({"behavior": test[test["Pred_Credit_Score"] == "Good"]["Payment_Behaviour"], "Credit_Score": poor_credit["Pred_Credit_Score"]})

plt.title("Payment_Behaviour by Credit Score", fontsize=20)
sns.countplot(data=test, y="Payment_Behaviour", hue="Pred_Credit_Score")
plt.show()

plt.title("Payment Behavior of Good Credit Score", fontsize=20)
sns.countplot(data=goodbehav, y="behavior")
plt.show()

standardbehav = pd.DataFrame({"behavior": test[test["Pred_Credit_Score"] == "Standard"]["Payment_Behaviour"], "Credit_Score": poor_credit["Pred_Credit_Score"]})

plt.title("Payment Behavior of Standard Credit Score", fontsize=20)
sns.countplot(data=standardbehav, y="behavior")
plt.show()

poorbehav = pd.DataFrame({"behavior": test[test["Pred_Credit_Score"] == "Poor"]["Payment_Behaviour"], "Credit_Score": poor_credit["Pred_Credit_Score"]})

plt.title("Payment Behavior of Poor Credit Score", fontsize=20)
sns.countplot(data=poorbehav, y="behavior")
plt.show()

plt.title("Payment of Min Amount by Prediction", fontsize=20)
sns.countplot(data=test, y="Payment_of_Min_Amount", hue="Pred_Credit_Score")
plt.show()

goodmin = pd.DataFrame({"Min": test[test["Pred_Credit_Score"] == "Good"]["Payment_of_Min_Amount"], "Credit_Score": poor_credit["Pred_Credit_Score"]})

plt.title("Payment of Minimum Amount of Good Credit Score", fontsize=20)
sns.countplot(data=goodmin, y="Min")
plt.show()

standardmin = pd.DataFrame({"Min": test[test["Pred_Credit_Score"] == "Standard"]["Payment_of_Min_Amount"], "Credit_Score": poor_credit["Pred_Credit_Score"]})

plt.title("Payment of Minimum Amount of Standard Credit Score", fontsize=20)
sns.countplot(data=standardmin, y="Min")
plt.show()

poormin = pd.DataFrame({"Min": test[test["Pred_Credit_Score"] == "Poor"]["Payment_of_Min_Amount"], "Credit_Score": poor_credit["Pred_Credit_Score"]})

plt.title("Payment of Minimum Amount of Poor Credit Score", fontsize=20)
sns.countplot(data=poormin, y="Min")
plt.show()

sns.histplot(data=test, x="Total_EMI_per_month", kde = True, hue="Pred_Credit_Score")
plt.title("Total EMI per month", fontsize=20)
plt.xlim(0,500)
plt.show()

plt.title("Outstanding Debt", fontsize=20)
sns.histplot(data=test, x="Outstanding_Debt", kde = True, hue="Pred_Credit_Score")
plt.show()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)

sns.set(rc={'figure.figsize':(15,8)})
sns.heatmap(cm,annot=True,cmap='YlOrRd')
plt.show()

from sklearn.metrics import accuracy_score
acc_score=accuracy_score(y_test,y_pred)
print(f'Accuracy of the Model: {(acc_score*100)//1} %')

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

def model_fit():
    scoring = 'accuracy'
    print("Cross Validation Score of Each Model\n")
    for i in range(len(clf)):
        score = cross_val_score(clf[i], x, y, cv=k_fold, n_jobs=1, scoring=scoring)
        print(model[i],":",round(np.mean(score)*100,2))
        
model_fit()

plt.title("Can I borrow money from the banks?", fontsize=20)
sns.countplot(x="Pred_Credit_Score", data=test)
plt.show()

print("Number of people who can't borrow money from bank    : ", len(test[test["Pred_Credit_Score"] == "Poor"]))
print("Number of people who can borrow money from bank      : ", len(test[test["Pred_Credit_Score"] != "Poor"]))
print("Number of people who can borrow more money from bank : ", len(test[test["Pred_Credit_Score"] == "Good"]))

plt.title("By Age Range - Can I borrow money from the banks?", fontsize=20)
sns.countplot(data=test, x ="Age_Range", hue="Pred_Credit_Score")
plt.show()

plt.title("By Number of Credit Card - Can I borrow money from the banks?", fontsize=20)
sns.countplot(data=test, x ="Num_Credit_Card", hue="Pred_Credit_Score")
plt.show()
