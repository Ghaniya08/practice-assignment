# Question 1 - 
# Simple Linear Regression A student wants to understand whether the number of study hours affects the marks obtained in an examination

# a. Create a Pandas DataFrame containing the above data.
# b. Separate the data into the independent variable X and dependent variable y. 
# c. Create and train a Linear Regression model using Scikit-learn. 
# d. Predict the expected marks of a student who studies for 6.5 hours. 
# e. Display the coefficient and intercept of the trained model.
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Study Hours': [2, 3, 4, 5, 6, 7, 8],
    'Exam Score': [45, 50, 56, 62, 68, 74, 81]
}

# a. Create a Pandas DataFrame containing the above data.
df = pd.DataFrame(data)
# print(df)

# b. Separate the data into the independent variable X and dependent variable y. 
X = df[['Study Hours']]
y = df['Exam Score']

# c. Create and train a Linear Regression model using Scikit-learn. 
model = LinearRegression()
model.fit(X, y)

# d. Predict the expected marks of a student who studies for 6.5 hours. 
predicted_score = model.predict([[6.5]])
print("Predicted Score:", predicted_score[0])

# e. Display the coefficient and intercept of the trained model.
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)