# Question 2 - Multiple Linear Regression
# A company wants to predict an employee's monthly performance score based on Training Hours, Years
# of Experience, and Number of Completed Projects.
# Use the following dataset:

# Tasks
# a. Create the dataset using Pandas. ✔
# b. Define the independent variables and target variable. ✔
# c. Train a Multiple Linear Regression model. ✔
# d. Predict the performance score for an employee with Training Hours = 14, Experience = 4 years, and
# Completed Projects = 6. ✔
# e. Display the coefficients of all three independent variables. ✔

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = {
    "Training Hours": [5, 8, 10, 12, 15, 18, 20, 22],
    "Experience (Years)": [1, 2, 3, 4, 6, 7, 8, 9],
    "Completed Projects": [2, 3, 4, 5, 6, 7, 8, 9],
    "Performance Score": [55, 62, 68, 74, 80, 85, 90, 94]
}

df = pd.DataFrame(dataset)
#  Define the independent variables and target variable
X = df[["Training Hours", "Experience (Years)", "Completed Projects"]]
y = df["Performance Score"]

# split the dataset into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#  Train a Multiple Linear Regression model.
model = LinearRegression()
model.fit(x_train, y_train)


#  Predict the performance score for an employee with Training Hours = 14, Experience = 4 years, and
# Completed Projects = 6.

new_employee = pd.DataFrame({
    'Training Hours' : [14],
    'Experience (Years)' : [4],
    'Completed Projects' : [6]
})

predicted_score = model.predict(new_employee)
print(predicted_score[0])
# e. Display the coefficients of all three independent variables.

print(model.coef_)