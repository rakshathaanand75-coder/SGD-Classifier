# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Set initial weights (theta) to zero.
2.Compute Predictions: Calculate predictions using the sigmoid function on the weighted inputs.
3.Calculate Cost: Compute the cost using the cross-entropy loss function.
4.Update Weights: Adjust weights by subtracting the gradient of the cost with respect to each weight.
5.Repeat: Repeat steps 2–4 for a set number of iterations or until convergence is achieved.
 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: RAKSHATHA S A
RegisterNumber:  212225220079

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
cr = classification_report(y_test, y_pred)
print("\nClassification Report:\n", cr)
*/
```

## Output:
<img width="588" height="355" alt="image" src="https://github.com/user-attachments/assets/9adc4ab1-0a6d-4e8a-bd80-948f96199717" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
