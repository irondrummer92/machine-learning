# Random to intialize random perceptron weights (if required)
# Numpy for vectors,matrices and dot products
import random as rd
import numpy as np


# Defining a perceptron

class perceptron(object):
    
    def __init__(self, num_inputs, weight_vector = None):
        # Defining the initial perceptron
        self.num_inputs = num_inputs
        weight_count = self.num_inputs + 1
        if(weight_vector == None):
            self.weights = [random.random() for i in range(weight_count)]
        else:
            if(len(weight_vector) != weight_count):
                raise Exception("Weight vector size invalid. Must be #Inputs + 1")
            else:
                self.weights = weight_vector
        self.weights = np.array(self.weights)
        self.output = None
    
    def computePerceptronOutput(self, input_vector):
        # Computes the output as a combination of inputs
        if(type(input_vector) is np.array):
            print "Input not a numpy array"
        else:
            if((self.weights.size - 1) == input_vector.size):   
                output_signal = np.dot(self.weights[1:], input_vector) + self.weights[0]
                self.output = 1 if (output_signal > 0) else 0
            else:
                print "Inputs provided is of invalide size"    
   
# -------------------------------------------------------------------------------------------------------------#
# Perceptron to simulate an AND gate
new_perceptron = perceptron(num_inputs = 2, weight_vector = [-0.8,0.5,0.5])

new_perceptron.computePerceptronOutput(np.array([0,0]))
new_perceptron.output
