from pyscript import document
output_div = document.querySelector("#textarea")


import pandas
from sklearn import linear_model
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
userinput = []
user2input = []
def submitbtn(event):
  for i in range(2,15):
    if (document.querySelector("#d"+str(i)).checked == True):
      userinput.append(1)
    else:
      userinput.append(0)
    user2input.append(document.querySelector("#d"+str(i)).checked)
      
  data = {
    "GENDER": [document.querySelector("#storeage").value],
    "AGE": [document.querySelector("#ageinput").value],
    "SMOKING": [userinput[0]],
    "YELLOW_FINGERS": [userinput[1]],
    "ANXIETY": [userinput[2]],
    "PEER_PRESSURE": [userinput[3]],
    "CHRONIC DISEASE": [userinput[4]],
    "FATIGUE ": [userinput[5]],
    "ALLERGY ": [userinput[6]],
    "WHEEZING": [userinput[7]],
    "ALCOHOL CONSUMING": [userinput[8]],
    "COUGHING": [userinput[9]],
    "SHORTNESS OF BREATH": [userinput[10]],
    "SWALLOWING DIFFICULTY" :[userinput[11]],
    "CHEST PAIN": [userinput[12]]
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
  riskindex = result[1].reshape(-1)[1]
  plhld1 = "Bootstrap Aggregation - Decision Tree Model\nDiagnosis: " + str(result[0]) +"\nRisk: " + str(riskindex*100) + "%\nAccuracy: " + str(oob_model.oob_score_)


  #Support Vector Machine
  from sklearn.model_selection import RandomizedSearchCV
  from sklearn.metrics import classification_report
  from sklearn.svm import SVC

  test_for_best_param={'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.001,0.01,0.1,1,10,100]}
  svm=RandomizedSearchCV(SVC(),test_for_best_param,cv=5)
  svm.fit(X,y)
  result = svm.predict(input_data)
  plhld2 = "\n\nSupport Vector Machine Model\nDiagnosis: "+ str(result)+"\nSVM model best parameters: "+ str(svm.best_params_)
  plhld3 = ""
  if (riskindex < 0.4):
    plhld3 = "LOW"
  elif (riskindex < 0.6):
    plhld3 = "MEDIUM"
  elif (riskindex < 0.8):
    plhld3 = "HIGH"
  else:
    plhld3 = "VERY HIGH"

  output_div.innerText = str(plhld1) + str(plhld2) + "\n\nYour risk of lung cancer is " + plhld3
  #str(data)+str(user2input)