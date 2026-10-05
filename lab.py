import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

data=pd.read_csv("spam.csv")
data['Category']=data['Category'].map({'ham':0,'spam':1})

X=data['Message']
Y=data['Category']

vectorizer=TfidfVectorizer(stop_words='english')
X=vectorizer.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=SVC(kernel='linear')
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

print("\nAccuracy: ",accuracy_score(Y_test,Y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))


