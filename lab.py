import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_csv("data.csv")
print(df.head())

X=df[["bedrooms,bathrooms,sqft_living,floors,view,yr_built"]]
y=["price"]
X_train,X_test,y_test,y_train=train_test_split(X,y,test_size=0.2,random_State=42)

model=LinearRegression()
model.fit(X_train,X_test)
y_pred=model.predict(X_test)