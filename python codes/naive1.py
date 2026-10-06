import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data=pd.read_csv("heart.csv")
X=data.drop("target",axis=1)

data.head()

Y=data["target"]

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
model=GaussianNB()
model.fit(X_train, Y_train)                                                                                                                     
Y_pred=model.predict(X_test)

print("Accuracy: ",accuracy_score(Y_test,Y_pred))
print("Confusion Matrix:\n", 
confusion_matrix(Y_test, Y_pred)) 
print("Classification Report:\n", 
classification_report(Y_test, Y_pred))

new_passenger=pd.DataFrame({
    'age':[40],
    'sex':[1],
    'cp':[0],
    'trestbps':[125],
    'chol':[250],
    'fbs':[0],
    'restecg':[1],
    'thalach':[150],
    'exang':[0],
    'oldpeak':[2.6],
    'slope':[2],
    'ca':[2],
    'thal':[3]
})

predicted_survival=model.predict(new_passenger)
print("Heart disease Prediction ",predicted_survival[0])

if(predicted_survival[0]==1):
    print("Has heart disease")
else:
    print("does not have heart disease")










