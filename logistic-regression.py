# Question 3 - 
# Logistic Regression A bank wants to predict whether a customer will be approved for a loan.
# a. Create the dataset in a Pandas DataFrame.
# b. Separate the features and target variable.
# c. Train a Logistic Regression classification model.
# d. Predict whether a customer with Income = Rs. 48,000 and Credit Score = 640 will receive a loan.
# e. Display the predicted class and prediction probability.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = {
    'Income': [30000, 35000, 40000, 45000, 50000,
               55000, 60000, 65000, 70000, 80000],

    'Credit Score': [550, 580, 600, 620, 650,
                     670, 700, 720, 750, 780],

    'Loan Approved': [0, 0, 0, 0, 1,
                      1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# method 1 
# X = df[['Income', 'Credit Score']]

# y = df['Loan Approved']

# method 2 
### Split dataset into independent and dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#  Train a Logistic Regression classification model.

model = LogisticRegression()
model.fit(X,y)

# Predict whether a customer with Income = Rs. 48,000 and Credit Score = 640 will receive a loan.

new_customer = pd.DataFrame({
    'Income':[48000],
    'Credit Score' : [640],

})

pred = model.predict(new_customer)
print(pred[0])
probability = model.predict_proba(new_customer)
print(probability)