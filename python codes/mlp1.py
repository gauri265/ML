import tensorflow as tf
from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Dense,Flatten
from  tensorflow.keras.datasets import mnist

(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train/255.0
X_test=X_test/255.0
model=Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.1)

loss,accuracy=model.evaluate(X_test, y_test)
print("Test Accuracy:",accuracy)
print("Test Loss:",loss)
import numpy as np
sample=X_test[0]
prediction=model.predict(np.expand_dims(sample,axis=0))
print("Actual Digit: ", y_test[0])
print("Predicted Digit: ", np.argmax(prediction))































