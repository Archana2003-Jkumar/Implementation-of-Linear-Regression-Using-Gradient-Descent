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

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function of using theta as parameter
   in a linear regression model
  """
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2
  return 1/(2*m) * np.sum(square_err) #returning 
  
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X,y and theta and update theta by taking num_iters gradient steps 
  with learning rate of alpha
  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) = "+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict(x,theta):
  """
  Takes in numpy array of x and theta and return the predicted value of y based on theta
  """
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
### Profit Prediction graph
![image](https://user-images.githubusercontent.com/93427594/235610622-409dbfef-7e35-4447-b024-a67ef88891fa.png)
### Compute Cost Value
![image](https://user-images.githubusercontent.com/93427594/235610744-d7064657-d18d-428d-89d6-61ee0acaefba.png)
### h(x) Value
![image](https://user-images.githubusercontent.com/93427594/235610825-f2d4e44f-5b4d-49d2-945a-ec94240b4118.png)
### Cost function using Gradient Descent Graph
![image](https://user-images.githubusercontent.com/93427594/235610891-2aaad5e9-4ca5-4701-abdc-793b839ea0b7.png)
### Profit Prediction Graph
![image](https://user-images.githubusercontent.com/93427594/235611203-9ecca8cb-11d9-44a1-9189-d4a800c0bea0.png)
### Profit for the Population 35,000
![image](https://user-images.githubusercontent.com/93427594/235611260-ce342a12-d7ba-44b6-925f-79852d8d57d6.png)
### Profit for the Population 70,000
![image](https://user-images.githubusercontent.com/93427594/235611424-13f271c1-b9ea-4863-9e61-460c9fb2b3fe.png)

 
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
