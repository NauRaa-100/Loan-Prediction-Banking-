"""
-- Loan Prediction Problem Dataset 

"""

#----------Importing Libraries----------

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier ,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression ,RidgeClassifier ,SGDClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix,f1_score,precision_score ,classification_report ,roc_curve, roc_auc_score
import joblib
#---------------Load Data--------------
df_=pd.read_csv(r'E:\Rev-DataScience\AI-ML\MLProjects_Structured_Data\loan.csv', encoding='latin')

#Info

print(df_.head(5))

#Check Empty values
print(df_.isnull().sum().sum()) #149 null values

#check Duplicated Values
print(df_.duplicated().sum()) # zero duplicated values

print(df_.shape) #614 , 13
print(df_.dtypes)

df_=df_.drop('Loan_ID',axis=1)

#print(df_.memory_usage(deep=True)) 

for col in df_.columns:
    if df_[col].dtype =='object':
        df_[col]=df_[col].astype('category')
        df_[col]=df_[col].fillna(df_[col].mode()[0])


    elif df_[col].dtype =='int64':
        df_[col]=df_[col].astype('int8')
        df_[col]=df_[col].fillna(df_[col].median())
    else:
        df_[col]=df_[col].astype('float32')
        df_[col]=df_[col].fillna(df_[col].median())


#Check Memory size and Null Values 

print(df_.memory_usage(deep=True)) 
print(df_.isnull().sum())

#------------Outliers---------------

def find_outliers(series):
    Q1= series.quantile(0.25)
    Q3=series.quantile(0.75)
    IQR= Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series < lower)| (series > upper)]

#Extracting numerical columns 

numerical=[]

for col in df_.select_dtypes(include=[np.number]).columns:
    numerical.append(col)

#Outliers in columns 

for col in df_[numerical]:
    print(f"-----Outliers in {col}-----")
    print(find_outliers(df_[col]).shape[0])
    
    plt.boxplot(df_[col])
    plt.title(f"Outliers in {col}")
    plt.show()

#----------

#Mapping The target Column 
#print(df_['Loan_Status'].unique())
df_['Loan_Status']=df_['Loan_Status'].map({"N":0,'Y':1})
df_['Loan_Status']=df_['Loan_Status'].astype('int8')

#--------------EDA & Relations----------------

for col in df_.columns:
    if df_[col].dtype == 'object' or str(df_[col].dtype).startswith('category'):
        print(f'-----Relation with {col}-----')
        print('--------------------------------')
        print(df_.groupby(col)['Loan_Status'].mean())
        print('------------------------------------')

        #---- Visualization of Relations ----

        plt.figure(figsize=(8,6))
        sns.barplot(x=col,y='Loan_Status',data=df_)
        plt.title(f'Relations between {col} and Loan Status')
        plt.show()

print(df_['Loan_Status'].value_counts())

#------

#Visualization of Loan Status Count 

plt.figure(figsize=(8,6))
sns.countplot(x='Loan_Status',data=df_)
plt.title("Distribution of Loan Status 1 = Yes , 0 = No")
plt.show()

"""
Overview Insights :-

**shape =>

-- Rows = 614 , Columns = 13 -- 12 After removing a column Loan_ID

**Empty Values =>

-- 149 Null values

**Duplicated values =>
-- Zero value

**Dtypes =>
-- Object(8) , floats64 (4) , int64(1)

Outliers =>
-- there are 4 columns have outliers

## Actions

-- Deleted not useful column 'Loan_ID'
-- Optimized Data with category , float32 , int8
-- Filled Null values with categorical columns--> Mode , Numerical Columns --> Median
-- Mapping the target column Yes : 1 , No : 0
-- Most Repeated Yes and No Loan status Yes = 422 , No = 192

"""


#========================================
# Encoding
#========================================


df=pd.get_dummies(df_,drop_first=True)


#========================================
# Scaling & Split
#========================================

x=df.drop('Loan_Status',axis=1).values
y=df['Loan_Status'].values

scaler=RobustScaler()
x_scaled=scaler.fit_transform(x)


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

#========================================
# Features Selection 
#======================================== 

select=RandomForestClassifier(n_estimators=100,random_state=42)
select.fit(x_train,y_train)
df=df.drop('Loan_Status',axis=1)
importances=pd.Series(select.feature_importances_,index=df.columns).sort_values(ascending=False)

#Top 5 features 
features=importances.head(5).index.to_list()

plt.figure(figsize=(8,5))
sns.barplot(x=importances.values[:5],y=importances.index.values[:5])
plt.title("Top Features")
plt.show()

#========================================
# Choosing Model 
#======================================== 

models={
'Logistic':LogisticRegression(max_iter=200),
'SVM':SVC(kernel='rbf',gamma='scale',random_state=42),
'Gradient':GradientBoostingClassifier(random_state=42),
'Forest':RandomForestClassifier(random_state=42,n_estimators=200),
'XGBoost':XGBClassifier(),
'Tree':DecisionTreeClassifier(max_depth=5,random_state=42),
'LightGBM':LGBMClassifier(random_state=42),
'CatB':CatBoostClassifier(verbose=0,random_state=42)


}

results=[]

for name , model in models.items():
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    f1=f1_score(y_test,pred)
    ps=precision_score(y_test,pred)

    results.append({'Model':name,'F1':f1,'Percision':ps})

results_df=pd.DataFrame(results).sort_values(by='F1',ascending=False)

print(results_df)

best_model=results_df.iloc[0]
print(best_model)

#Applying Model
model=LogisticRegression(max_iter=300)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_prob=model.predict_proba(x_test)[:,1]

acc=model.score(x_test,y_test)
clsr=classification_report(y_test,y_pred)

scores = cross_val_score(model, x_scaled, y, cv=5, scoring='f1')
print("Average F1:", scores.mean())

#Visualization of Roc curve

fpr,tpr,_=roc_curve(y_test,y_prob)
auc=roc_auc_score(y_test,y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f'AUC:{auc:.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.legend()
plt.title("Visualization of Roc Curve")
plt.show()


#Visualization of Confusion Matrix

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#General Relation
plt.figure(figsize=(9,6))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.title('Numerical Relations With Loan Status')
plt.show()




#=============================================
# Saving Model -- 
#=============================================

joblib.dump(model,'Model.pkl')
joblib.dump(scaler,'Scaler.pkl')
print('Done')

