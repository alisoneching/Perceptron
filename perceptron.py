#-------------------------------------------------------------------------
# AUTHOR: Alison Ching
# FILENAME: perceptron.py
# SPECIFICATION: train a Single Layer Perceptron and a Multi-Layer Perceptron to classify optically recognized handwritten digits
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

# track highest accuracy of the two models
highest_perceptron_accuracy = 0
highest_mlp_accuracy = 0

for rate in n: #iterates over n

    for bool in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here

        for model in ['Perceptron', 'MLP']: #iterates over the algorithms

            #Create a Neural Network classifier
            if model == 'Perceptron':
                clf = Perceptron(eta0 = rate, shuffle = bool, max_iter=1000)
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init = rate,
                                    hidden_layer_sizes = (25,),
                                    shuffle = bool, max_iter=1000)
            #-->add your Pyhton code here

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            correct_preds = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                if prediction == y_testSample:
                    correct_preds += 1
            accuracy = correct_preds / len(y_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here

            if model == 'Perceptron' and accuracy > highest_perceptron_accuracy:
                highest_perceptron_accuracy = accuracy
                print(f"Highest Perceptron accuracy so far: {accuracy:.2f}, Parameters: learning rate={rate}, shuffle={bool}")
            
            elif model == 'MLP' and accuracy > highest_mlp_accuracy:
                highest_mlp_accuracy = accuracy
                print(f"Highest MLP accuracy so far: {accuracy:.2f}, Parameters: learning rate={rate}, shuffle={bool}")
    










