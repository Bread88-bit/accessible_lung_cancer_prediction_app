import pandas
from sklearn import linear_model
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


data = {
  "GENDER": [1],
  "AGE": [70],
  "SMOKING": [1],
  "YELLOW_FINGERS": [1],
  "ANXIETY": [1],
  "PEER_PRESSURE": [1],
  "CHRONIC DISEASE": [0],
  "FATIGUE ": [0],
  "ALLERGY ": [0],
  "WHEEZING": [1],
  "ALCOHOL CONSUMING": [1],
  "COUGHING": [1],
  "SHORTNESS OF BREATH": [1],
  "SWALLOWING DIFFICULTY" :[1],
  "CHEST PAIN": [0]
}

input_data = pandas.DataFrame(data)



df = pandas.read_csv("lungcancerdataset.csv")
df.drop_duplicates(inplace=True)

from sklearn.preprocessing import LabelEncoder
encoda = LabelEncoder()
df['LUNG_CANCER']=encoda.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoda.fit_transform(df['GENDER'])

X=df.drop(['LUNG_CANCER'],axis=1)
y=df['LUNG_CANCER']
e = {1:0,2:1}
for j in range(2,len(X.columns)):
    X[X.columns[j]] = X[X.columns[j]].map(e)



#start modelling
from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X['AGE']=scaler.fit_transform(X[['AGE']])
input_data['AGE']=scaler.transform(input_data[['AGE']])


#Decision Tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

features = X.columns

#Bootstrap Aggregation -- Decision Tree
#use n_estimation = 30 for randomstate = 42

#out-of-bag metric -- Improving Decision Tree
from sklearn.ensemble import BaggingClassifier

oob_model = BaggingClassifier(n_estimators = 100, oob_score = True,random_state = 42)
oob_model.fit(X, y)

result = [oob_model.predict(input_data),oob_model.predict_proba(input_data)]
print("Bootstrap Aggregation - Decision Tree Model\nDiagnosis:",result[0],"\nIndex:",result[1].reshape(-1)[1],"\nAccuracy:",oob_model.oob_score_)


#Support Vector Machine
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

test_for_best_param={'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.001,0.01,0.1,1,10,100]}
svm=RandomizedSearchCV(SVC(),test_for_best_param,cv=5)
svm.fit(X,y)
result = svm.predict(input_data)
print("\n\nSupport Vector Machine Model\nDiagnosis:",result,"\nSVM model best parameters: ", svm.best_params_)