import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv("weather.csv")

x=data[['Day']]

y=data['Temperature']

model=LinearRegression()

model.fit(x,y)

prediction=model.predict([[11]])

print("Predicted temperature on Day 11:",prediction[0])

plt.scatter(x,y,color='blue',
            label='Actual Data')
plt.plot(x, model.predict(x),
            color='red', label='prediction Line')

plt.title("Wheather Data Analysis and Prediction")
plt.xlabel("Day")
plt.ylabel("Temperature(°c)")
plt.legend()
plt.show()




