import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv")
data.head()

X = data[["bedrooms","bathrooms","sqft_living","floors","condition", "waterfront", "view", "yr_built"]]
y= data["price"]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

print("R2 scores", r2_score(y_test, y_pred))
print("MAE", mean_absolute_error(y_test, y_pred))
print("MSE", mean_squared_error(y_test, y_pred))
print("Intercept", model.intercept_)

new_house = pd.DataFrame({
    'bedrooms':[3],
    'bathrooms':[3],
    'sqft_living':[2600],
    'floors':[2],
    'condition':[3],
    'waterfront':[0],
    'view':[1],
    'yr_built':[9],
})
predicted_price=model.predict(new_house)
print("Predicted Value: ",predicted_price)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs Predicted prices")
plt.show()


















