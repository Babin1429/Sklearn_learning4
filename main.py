
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = load_iris()
x = data.data
y = data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
scalin=StandardScaler()
x_train_scaled= scalin.fit_transform(x_train)
x_test_scaled= scalin.transform(x_test)
#print(x_train_scaled)
imputer=SimpleImputer(strategy="mean")
x=imputer.fit_transform(x)
#print(x)
Pipe_line=Pipeline([
    ("scaler",StandardScaler)
    ("Model",LogisticRegression)
])
Pipeline.fit(x_train,x_test)
Pipeline.score(x_test,y_test)

randomfrst= RandomForestClassifier()
r1=randomfrst.score(x_test,y_test)
r2=randomfrst.fit(x_train,y_train)
print(r1)
print(r2)