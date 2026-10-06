import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

data=pd.read_csv("train.csv")

features=["Pclass","Sex","Age","SibSp","Parch","Fare"]
data=data[features+['Survived']]

data['Sex']=data['Sex'].map({"male":0,"female":1})

data=data.dropna()

X=data[["Pclass","Sex","Age","SibSp","Parch","Fare"]]
y=data["Survived"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Accuracy: ",accuracy_score(y_test,y_pred))

new_passenger=pd.DataFrame({
    'Pclass':[3],
    'Sex':[0],
    'Age':[20],
    'SibSp':[1],
    'Parch':[1],
    'Fare':[50]
})

predicted_survival=model.predict(new_passenger)
print("Survival: ",predicted_survival[0])

if(predicted_survival[0]==1):
    print("Passeenger had survived")
else:
    print("Passenger had not survived")

















