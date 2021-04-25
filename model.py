# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:24:06 2021

@author: USER
"""
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('titanic_train.csv')
df.drop('Cabin',axis=1,inplace=True)
df['Age']=df['Age'].fillna(df['Age'].mode()[0])
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
sex_mapping={'male':1,'female':2}
df['Sex']=df['Sex'].map(sex_mapping)
embark_mapping={'S':1,'C':2,'Q':3}
df['Embarked']=df['Embarked'].map(embark_mapping)
df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
x=df.drop('Survived',axis=1)
y=df.Survived
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rfc_classifier = RandomForestClassifier()
rfc_classifier.fit(x_train, y_train)

pickle.dump(rfc_classifier,open('titanic_classifier.pkl','wb'))
model=pickle.load(open('titanic_classifier.pkl','rb'))
print(model.predict([[1,2,38.0,1,0,71.2833,2]]))


