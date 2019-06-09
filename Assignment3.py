# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(6)
X = np.random.rand(3,4)*10
np.random.seed(101)
Y = np.array([[1], [1], [0]])
print(X)
print(Y)
def Sigmoid(z) :
    return 1/(1+np.exp(-z))         
def Derivative_sigmoid(z):
    return (Sigmoid(z)*(1-Sigmoid(z)))
epochs = 3000
learning_rate = 0.01
print("The number of hidden layers : 2")
print("The number neurons input layer is 4")
print("The number of neurons in output layer is 3")
m = Y.shape[0]
hidden_layer1 = 5
hidden_layer2 = 3
input_layer = X.shape[1]
output_layer = Y.shape[1]
print("Size of first hidden layer : " , hidden_layer1 )
print("Size of second hidden layer : " , hidden_layer2 )
theta1 = np.random.rand(hidden_layer1 , input_layer+1)*2-1
theta2 = np.random.rand(hidden_layer2 , hidden_layer1+1)*2-1
theta3 = np.random.rand(output_layer , hidden_layer2+1)*2-1
def Cost_function() :
    # Feed forward part for computing activatins and cost
    a1 = np.hstack((np.ones((m,1)),X))
    z2 = np.dot(a1, theta1.T)
    a2 = Sigmoid(z2)
    a2= np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2 , theta2.T)
    a3 = Sigmoid(z3)
    a3 = np.hstack((np.ones((m,1)), a3))
    z4 = np.dot(a3 , theta3.T)
    h = Sigmoid(z4) #Value of output neuron
    cost = 1/m*(np.sum(np.multiply(Y, np.log(1-h))+np.multiply(1-Y, np.log(h))))
    # Back prop part for finding out gradients
    delta4 =  h-Y
    delta3 = np.dot(delta4 , theta3)*Derivative_sigmoid(np.hstack((np.ones((m,1)),z3)))
    delta3 = delta3[:,1:]
    delta2 = np.dot(delta3, theta2)*Derivative_sigmoid(np.hstack((np.ones((m,1)),z2)))
    delta2 = delta2[:,1:]
    theta1_grad = 1/m*(np.dot(delta2.T , a1))
    theta2_grad = 1/m*(np.dot(delta3.T , a2))
    theta3_grad = 1/m*(np.dot(delta4.T , a3))
    return cost , h , theta1_grad , theta2_grad ,theta3_grad 
for epoch in range(epochs) :
    #Gradient descent Step
    cost,h,theta1_grad,theta2_grad,theta3_grad =Cost_function()
    theta1 = theta1 - learning_rate*theta1_grad
    theta2 = theta2 - learning_rate*theta2_grad
    theta3 = theta3 - learning_rate*theta3_grad
    if((epoch+1)%50==0) :
        print("Epoch : " , epoch+1 , " Cost = " , cost , "Values of output nueron :" ,h )
Trained_var = Cost_function()
y_pred = Trained_var[1]>=0.5
y_pred = y_pred.astype(int)
print("The output vsluees predicted from neural net : " , y_pred)
print("Actual Output values: " , Y)
Accuracy = 0
for i in range(m):
    if(y_pred[i]==Y[i]) : Accuracy +=1
Accuracy = Accuracy*100/m
print("Accuracy of trained neural net : " , Accuracy , "%")