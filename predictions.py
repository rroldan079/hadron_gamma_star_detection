import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

'''
Types of predictions
--------------------

Supervised Learning predictions
Classification- predict discrete classes
ex. identify hot dogs, pizza, ice cream, etc (can be labeled)

binary classification- hot dog or not hotdog

classifying two things, states, etc. (two of anything)

Multi class classification- more than two


Regression- predict continuous values
ex. price or etherium, temperature, price of homes, something that is scaled

Predicting a number closest to a true value as possible


Questions to consider:

How to make the model learn?
How to tell if it is learning?


Supervised Learning Data Sets consist of:

rows are know as a sample
columns are know as features
outcomes are know as output labels

all values within a row (excluding the output label) are feature vectors
output of a feature vector is called the target 
everything included in the data set (exclusing the output label) is called the feature matrix (X)


How do models work?
        
                    training: when you take the value or prediction, compare it to the true value 
                    in the data set, and send it back to the model to reduce the error (loss)

           ----- <------------
Input --> |Model| --> Output |
           -----


           
Can our model handle new data?

Break up data set into three different type


Training dataset
Validation dataset
Testing dataset



Feed training data set into the model, then determine difference between true value and the prediction, and reduce the loss through training

After model is adjusted, Validation set is used as a reality check during/after training to ensure model and handle unseen data

Validation testing does not have a closed feedback loop, so loss is never corrected

Higher the loss, the less accurate the model


Testing set used to check how generalizable the final chosen model is
Used when model tested has gotten to the lowest loss it could possibly get

output is the final reported performance of the model
  

Loss is the difference between the prediction and true value


L1 Loss

Loss = sum(|y_real - y_predicted|)

Can be represented as a function

Used to handle outliers and noisy data
Compressive sensing and sparse modeling
Feature selection and dimensionality reduction


L2 Loss

Loss = sum((y_real - y_predicted)^2)

Solutions are smoother since it penalizes errors more heavily

Mostly used for things like linear regression, Guassian noise modeling and denoising
neural networks and deep learning


Binary cross-entropy loss (for binary classification)

loss = -1/(N * sum(y_real * log(y_predicted) + (1-y_real) * log((1-y_predicted))))

(loss decreases as the performance gets better)

Metrics of performance
---------------------
Accuracy of the model

'''

# these are features that allow the model to identify the class- this is an example on supervised learning
col_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym","fM3Long", "fM3Trans", "fAlpha", "fDist","class"]
csv_data = pd.read_csv("magic+gamma+telescope/magic04.data", names=col_names)

# represents values as integer values (either 1 or 0)
csv_data["class"] = (csv_data["class"] == "g").astype(int)

# function creates histograms using the csv data, plotting out the gamma and hadron stares.
def create_hist_head():
    print(csv_data.head())

    # plotting data on histogram for every feature and its data in the feature matrix
    for label in col_names[:-1]:
        plt.hist(csv_data[csv_data["class"] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
        plt.hist(csv_data[csv_data["class"] == 0][label], color='red', label='hadron', alpha=0.7, density=True)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()
        plt.show()



# creating train, validation, and test datasets

# split array into subarrays for training, validation, and testing data sets
# note that this method of splitting is deprecated and will be removed at some point- look into iloc instead
train, valid, test = np.split(csv_data.sample(frac=1), [int(0.6 * len(csv_data)), int(0.8 * len(csv_data))])
create_hist_head()

'''
scale_dataset() takes dataframe and an oversample boolean
as arguments


Scale dataset returns data as a horizontal stack, along with it's respective x and y values (equal amounts of x/y values )

'''
def scale_dataset(dataframe, oversample=False):
    x_values = dataframe[dataframe.columns[:-1]].values
    y_values = dataframe[dataframe.columns[-1]].values
    
    scaler = StandardScaler()
    x_values = scaler.fit_transform(x_values)

    if oversample:
        ros = RandomOverSampler()
        x_values, y_values = ros.fit_resample(x_values, y_values)

    data = np.hstack((x_values, np.reshape(y_values, (-1,1))))

    return data, x_values, y_values

# datasets finally prepared!
train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample= False)



# different types of models

'''
K-nearest neighbors
------------------

Ex. # of kids (y) as a function of income (x)

Defining distance and classify several points

+ = own a car
- = no car


Compare what's around a predicted point, and take the label of the majority of points near that predicted point

distance function to look at these points is known as Euclidean distance (straight line distance) between the predicted point and the points around it

k represents the number of neighbors to conclude the label to use (3-5 neighbors normally), so take the closest 3-5 points to the label as a way of predicting the pattern
'''


#ex. of implementing K-nearest neighbors model


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)

print(classification_report(y_test, y_pred))



# implementation of guassian naive bayes model

nbt_model = GaussianNB()
nbt_model = nbt_model.fit(x_train, y_train)

nb_y_pred = nbt_model.predict(x_test)

print(classification_report(y_test, nb_y_pred))


# implementation of logistic regression 
lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train, y_train)
lg_model_predict = lg_model.predict(x_test)

print(classification_report(y_test, lg_model_predict))



# implementation of a support vector machine (SVM) model
svc_model = SVC()
svc_model = svc_model.fit(x_train, y_train)
svc_model_predict = svc_model.predict(x_test)

print(classification_report(y_test, svc_model_predict))

'''
Naive Bayes
-----------

Understanding conditional probability + bayes' rule

Bayes' rule:

P(A|B) = P(B|A) * P(A)/ P(B)



Bayes' rule expanded:

P(Ck|x) = P(x|Ck) * P(Ck)/P(x)

x = feature vector
Ck = category
P(x|Ck) = likelihood
P(Ck) = prior
evidence = P(x)

P(Ck| x1, x2, x3, ..., xn) is proportional to P(x1, x2, x3, ..., xn | Ck) * P(Ck)




Logistic Regression

Classifying based on Regression


Modeling probability using a linear equation

p/1-p = mx + b
 



sigmoid function
s(x) = 1/1 + e^-x



logistic regression is all about trying to fit data within a sigmoid function



simple logistic regression -> when only dealing with x 

multiple logistic regression - x0, x1, x2, ... xn





Support Vector Machines (SVM)
----------------------------

Finding a dividing line or plane (depending on dimensions), or hyperplane, that
divides different groups of data



Margin is something important in SVM- additionally, the lines of the two closest points from both groups-
the larger the margins, the better

Support vectors: data that lies on the margin line 



Project values if it is difficult to drawing a hyperplane to define the two groups
This is done by-
kernel trick: changing from x -> (x, x^2)



Neural Network
--------------

Neural Networks are composed of input layers, hidden layers, and an output layer


layers are composed of a neuron

features are inputted into the neural net

All these features get weighted by some value, and the sums of these products go into a neuron

bias term is used to shift the value of this sum and run that through an activation function; After applying the activation function, an output is returned

Neural nets are composed of networks of interconnected neuron layers


Without activation functions, these models become linear 

Models become known as:

Sigmoid functions
Tanh function
RELU funcition



L2 Loss function- Quadratic formula

Loss depends on the y, where y = 0 means no loss.

To reduce the loss off a model, you can use gradient descent to lower the value to the lowest value of the quadratic function


This is what happens during training.

That is called backpropogation


new_weight_value = old_weight_value + alpha * reducing_value


alpha is known as the learning rate, which controls how long it takes for neural net to converge

This domne multiple times until the loss is reduced to its lowest value
'''

