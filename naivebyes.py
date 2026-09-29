# Question 4 - Naive Bayes Classification
# A simple system is required to classify an email as Spam or Not Spam based on the presence of selected
# words.
# 1 means the word is present; 0 means the word is absent.    Spam = 1 means Spam; Spam = 0 means Not Spam.
# Tasks
# a. Create the dataset using Pandas.
# b. Separate the features Free, Offer, and Meeting from the target Spam.
# c. Train a Bernoulli Naive Bayes classifier.
# d. Predict the class of a new email with Free = 1, Offer = 1, and Meeting = 0.
# e. Write 2-3 lines explaining why Naive Bayes is called "Naive".

import pandas as pd
from sklearn.naive_bayes import BernoulliNB

# a. Create the dataset using Pandas.

data = {
    'Free': [1, 1, 1, 0, 0, 0, 0, 1],
    'Offer': [1, 1, 0, 1, 0, 0, 1, 0],
    'Meeting': [0, 0, 0, 0, 1, 1, 1, 1],
    'Spam': [1, 1, 1, 1, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# print(df)

# b. Separate the features Free, Offer, and Meeting from the target Spam.

X = df[['Free', 'Offer', 'Meeting']]

y = df['Spam']

# c. Train a Bernoulli Naive Bayes classifier.

model = BernoulliNB()
model.fit(X,y)

# d. Predict the class of a new email with Free = 1, Offer = 1, and Meeting = 0.

new_email = [[1, 1, 0]]
prediction = model.predict(new_email)

print("Predicted Class:", prediction[0])

# e. Write 2-3 lines explaining why Naive Bayes is called "Naive".

# Naive Bayes is called "naive" because it assumes that every single feature in a dataset is completely independent of the others