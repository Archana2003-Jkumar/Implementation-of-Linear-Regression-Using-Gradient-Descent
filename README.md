# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the required packages.
2. Display the output values using graphical representation tools as scatter plot and graph.
3. predict the values using predict() function.
4. Display the predicted values and end the program

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:J.Archana priya 
RegisterNumber:  212221230007
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("/content/ex1.txt",header = None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city(10,000s")
plt.ylabel("profit ($10,000")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err)#returning
  
  
  
  data_n = data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)#call function

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    preds = x.dot(theta)
    error = np.dot(x.transpose(),(preds -y))
    descent = alpha * 1/m * error
    theta-=descent
    j_history.append(computeCost(x,y,theta))
  return theta,j_history


theta,j_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" +"+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function")

def predict(x,theta):
  pred = np.dot(theta.transpose(),x)
  return pred[0]

pred1 = predict(np.array([1,3.5]),theta)*10000
print("Population = 35000 , we predict a profit of $"+str(round(pred1,0)))

pred2 = predict(np.array([1,7]),theta)*10000
print("Population = 70000 , we predict a profit of $"+str(round(pred2,0)))
```

## Output:
![Screenshot 2023-03-31 133122](https://user-images.githubusercontent.com/93427594/229069830-414f4a7b-9f97-4ada-8fcc-d20f3a9cc924.png)

![Screenshot 2023-03-31 133209](https://user-images.githubusercontent.com/93427594/229069863-ae2b5243-40d3-4f33-be13-0b12680fad44.png)

![Screenshot 2023-03-31 133229](https://user-images.githubusercontent.com/93427594/229069890-ac6c9568-27b6-4673-9d7d-5307fa45bc7e.png)

![Screenshot 2023-03-31 133309](https://user-images.githubusercontent.com/93427594/229069924-36678e5c-c596-41d5-96d7-19118a40d4f2.png)

![Screenshot 2023-03-31 133345](https://user-images.githubusercontent.com/93427594/229069951-21fb8178-8aee-4e60-bab2-3012032e1cdc.png)

![Screenshot 2023-03-31 133402](https://user-images.githubusercontent.com/93427594/229069969-f5d19f48-2c62-4acb-b638-bf5a17ce343c.png)

![Screenshot 2023-03-31 135430](https://user-images.githubusercontent.com/93427594/229070000-0e8647e8-728d-46f2-8ebd-4a1f9ca3ecce.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
