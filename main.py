import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# these are features that allow the model to identify the class- this is an example on supervised learning
col_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym","fM3Long", "fM3Trans", "fAlpha", "fDist","class"]
csv_data = pd.read_csv("magic+gamma+telescope/magic04.data", names=col_names)


print(csv_data.head())
print("")
print(csv_data["class"])
csv_data["class"] = (csv_data["class"] == "g").astype(int)
print(csv_data["class"].head())

# What is machine learning-
'''
Subdomain of computer science that focuses on algorithms which help a computer
learn from data without explicit programming



Artificial intelligence: area of computer science where the goal is to
enable computers/machines to perform human-like tasks and simulate human behavior


machine learning is a subset of AI that tries to solve a specific problem and make predictions
using certain data

Data Science: field that attempts to find patterns and draw insights from data
(might use ML!)

All fields overlap, all fields may use ML



Types of Machine learning models
---------------------------------

Supervised Learning:

Used labeled inputs (meaning that the input has a corresponding output label) to
train models and learn outputs


Unsupervised learning: uses unlabeled data to learn about patterns in the data



Reinforcement learning: agent learning in interactive environment based on awards and penalties

This is kind of how a human trains a dog to sit or do a high five, based on a system of awards (giving it a treat)




Regarding Supervised Learning:



Inputs ---> Model ---> Output (predictions)

Inputs are feature vectors

Features can be:

Qualitiative: categorical data (finite number of categories or groups)
example: gender, nations, etc.


No inherent order to these categorical data sets (known as nominal data)



One-Hot Encoding: if data matches some category, set it to 1 (true), else 0 (false)
[USA, India, Canada, France]

if input from US -> [1,0,0,0]
if input from India -> [0,1,0,0]
etc...


Ordinal data (data with an inherent order, which is a different type of qualitative feature)

ex. 1-5 ratings in a product or service



Quantitative- numerical valued data (could be discrete (integers) or continuous(real numbers))
'''


