import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_excel("clean.xlsx")
data.sort_values(by='JourneyDate',inplace=True,ascending=True)

#Pre-process for target variable
chart_type = data['ChartingStatus'].value_counts()
#Here since W/l and PQW/l are not confirmed
#Rest all other are confirmed
#Confirmed - 1
#Not-confirmed - 0

data['ChartingStatus'] = data.ChartingStatus.str.replace('W/L', '0').replace('PQW/L','0')
data['ChartingStatus'] = data.ChartingStatus.str.replace('[A-Za-z1-9\s]+', '1')
data['ChartingStatus'] = data.ChartingStatus.str.replace('10', '1')

#Now plot some countplot()
c1 = sns.countplot(x=data['ChartingStatus'],data=data)
c1 = sns.countplot(x=data['ChartingStatus'],hue=data['TrainNo'],data=data)
c1 = sns.countplot(x=data['ChartingStatus'],hue=data['ClassOfTravel'],data=data)
data['BookingStatus'].value_counts()

#Enocode the categorical variable
data = pd.get_dummies(data,columns=['BookingStatus','TrainNo','ClassOfTravel'])

#Feature engineering for Journeydate
column_1 = data['JourneyDate']

temp = pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              #"hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              #"dayofweek": column_1.dt.dayofweek,
              #"weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })

data.reset_index(drop=True, inplace=True)
temp.reset_index(drop=True, inplace=True)
data = pd.concat([data,temp],axis=1)

JourneyDate = data['JourneyDate']

target = data['ChartingStatus']
data.drop(columns=['ChartingStatus','JourneyDate'],axis=1,inplace=True)
#Now split the dataset into the train and test datset
from sklearn.model_selection import train_test_split
X_train , X_test , y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=29)

#Parameter Tunning for XGboost
from xgboost.sklearn import XGBClassifier  
import scipy.stats as st
from sklearn.metrics import explained_variance_score

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

xgbreg = XGBClassifier(nthreads=-1)  

from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(xgbreg, params, n_jobs=-1)  
gs.fit(X_train, y_train,verbose=True)  

predictions = gs.predict(X_test)
print(explained_variance_score(predictions,y_test))


