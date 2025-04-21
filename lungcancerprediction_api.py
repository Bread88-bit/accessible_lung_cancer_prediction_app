from pyscript import document

r_identities = ["#r_adj","#r_m1d","#r_m1r","#r_m1a","#r_m2d","#r_m2bp"]
output_div = []
for l in r_identities:
  output_div.append(document.querySelector(l))


import pandas
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

  from sklearn.preprocessing import StandardScaler
  scaler=StandardScaler()
  X['AGE']=scaler.fit_transform(X[['AGE']])
  input_data['AGE']=scaler.transform(input_data[['AGE']])


  #Decision Tree
  from sklearn import tree
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import accuracy_score


  features = X.columns

  #Bootstrap Aggregation -- Decision Tree
  #use n_estimation = 30 for randomstate = 42

  #out-of-bag metric -- Improving Decision Tree
  from sklearn.ensemble import BaggingClassifier

  oob_model = BaggingClassifier(n_estimators = 100, oob_score = True,random_state = 42)
  oob_model.fit(X, y)

  result = [oob_model.predict(input_data),oob_model.predict_proba(input_data)]
  riskindex = result[1].reshape(-1)[1]
  diagvar = [int(result[0][0])]

  output_div[1].innerHTML = "模型结果：" + str(result[0][0] == 1)
  output_div[2].innerHTML = "风险：" + str(riskindex*100) + "%"
  output_div[3].innerHTML = "准确度：" + str(oob_model.oob_score_)


  #Support Vector Machine
  from sklearn.model_selection import RandomizedSearchCV
  from sklearn.metrics import classification_report
  from sklearn.svm import SVC

  test_for_best_param={'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.001,0.01,0.1,1,10,100]}
  svm=RandomizedSearchCV(SVC(),test_for_best_param,cv=5)
  svm.fit(X,y)
  result = svm.predict(input_data)
  output_div[4].innerHTML = "模型结果：" + str(result[0] == 1)
  output_div[5].innerHTML = "最佳参数：" + str(svm.best_params_)  
  diagvar.append(int(result[0]))
  
  plhld3 = ""
  if (diagvar[0]+diagvar[1] == 2):
    plhld3 = "高"
  elif (diagvar[0]+diagvar[1] == 1):
    plhld3 = "中"
  elif (diagvar[0]+diagvar[1] == 0):
    plhld3 = "低"
  output_div[0].innerHTML = "您的风险是：" + plhld3 + "。"
