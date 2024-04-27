# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Get population and profit data for cities.
2. Begin with random guesses for how population influences profit.
3. Gradually adjust guesses to minimize the difference between predicted and actual profits.
4. Keep adjusting until predictions are close to actual profits.
5. Once adjusted, predict profit for new city populations.
6. Evaluate how well predictions match actual profits.
7. If predictions are off, refine guesses and repeat the process.
8. Once satisfied, use the model to predict profits based on population for decision-making.

## Program:
```py
Program to implement the linear regression using gradient descent.
Developed by: VIKAMUHAN REDDY.N
RegisterNumber: 212223240181

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
 X = np.c_[np.ones(len(X1)),X1]

 theta = np.zeros(X.shape[1]).reshape(-1,1)
 for _ in range(num_iters):
  #Calculate predictions
  predictions = (X).dot(theta).reshape(-1,1)
  #Calculate errors
  errors=(predictions-y).reshape(-1,1)
  #update theta using gradient descent
  theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
 return theta
data=pd.read_csv("50_Startups.csv")
data.head()

X = (data.iloc[1:,:-2].values)
X1 =X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
#predict target calue for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value:{pre}")


```

## Output:
### 1. data.head()
![WhatsApp Image 2024-04-27 at 11 36 38_9a5122ca](https://github.com/vikamuhan-reddy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144928933/4bf16cc9-829f-4a9e-b3ee-d407b59858f7)

### 2. X1_Scaled
![WhatsApp Image 2024-04-27 at 11 37 08_f2e31197](https://github.com/vikamuhan-reddy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144928933/41d3e74b-8261-42e5-b70c-dfdea5a47126)

### 3. predicted value
![WhatsApp Image 2024-04-27 at 11 37 36_a776e43f](https://github.com/vikamuhan-reddy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144928933/eafb7895-a0c6-4025-8016-2ed3a9628499)







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
