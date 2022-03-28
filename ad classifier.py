#importing necessary libraries

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#Visualizing the data

data = pd.read_csv("socialmedia.csv")
print(data.head())

print(data.describe())

print(data.isnull().sum())

#Analysing buying with respect to Age

plt.figure(figsize=(15, 10))
plt.title("Product Purchased By People Through Social Media Marketing")
sns.histplot(x="Age", hue="Purchased", data=data)
plt.show()

#Analysing buying with respect to Income

plt.title("Product Purchased By People According to Their Income")
sns.histplot(x="EstimatedSalary", hue="Purchased", data=data)
plt.show()

#Splitting the dataset

x = np.array(data[["Age", "EstimatedSalary"]])
y = np.array(data[["Purchased"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, 
                                                random_state=42)

#Building the Model

model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

predictions = model.predict(xtest)

#RESULT
print(classification_report(ytest, predictions))



