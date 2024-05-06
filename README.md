# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Start

step 1
1. Collect city population and profit data.
2. Make initial guesses on how population might affect profit.

step 2
Start loop:
1. Predict profits using initial guesses.
2. Compare predicted profits with actual profits.
3. Adjust guesses to minimize prediction error.
4. Check if prediction error is acceptable (e.g., within a certain threshold).
5. If prediction error is acceptable, stop loop.
6. If prediction error is not acceptable, go back to step 3a.

step 3
Once satisfied with the model:
1. Use the model to predict profits for new city populations.
Test the model's accuracy:
2. Collect new city population data.
3. Predict profits using the model.
4. Compare predicted profits with actual profits.

step 4
1. If prediction accuracy is satisfactory, stop.
2. If prediction accuracy is not satisfactory, refine the model further.

stop


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
![image](https://github.com/vikamuhan-reddy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144928933/86d433ca-b523-4dad-a291-53ee6a4fc5a8)
![image](https://github.com/vikamuhan-reddy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144928933/ab3f0406-7fb1-4898-8290-a66f710408fa)
![image](https://github.com/vikamuhan-reddy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144928933/b8d57ed3-9479-4c17-9807-b817dce43f89)



### 3. predicted value
![WhatsApp Image 2024-04-27 at 11 37 36_a776e43f](https://github.com/vikamuhan-reddy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144928933/eafb7895-a0c6-4025-8016-2ed3a9628499)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
