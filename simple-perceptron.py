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
# AND function can be simulated using w0 = -0.8, w1=w2 =0.5

and_perceptron = perceptron(num_inputs = 2, weight_vector = [-0.8,0.5,0.5])

# Testing all input combinations
and_perceptron.computePerceptronOutput(np.array([0,0]))
print and_perceptron.output
and_perceptron.computePerceptronOutput(np.array([0,1]))
print and_perceptron.output
and_perceptron.computePerceptronOutput(np.array([1,0]))
print and_perceptron.output
and_perceptron.computePerceptronOutput(np.array([1,1]))
print and_perceptron.output

# -------------------------------------------------------------------------------------------------------------#
# OR function can be simulated using w0 = -0.3, w1=w2 =0.5

or_perceptron = perceptron(num_inputs = 2, weight_vector = [-0.3,0.5,0.5])

# Testing all input combinations
or_perceptron.computePerceptronOutput(np.array([0,0]))
print or_perceptron.output
or_perceptron.computePerceptronOutput(np.array([0,1]))
print or_perceptron.output
or_perceptron.computePerceptronOutput(np.array([1,0]))
print or_perceptron.output
or_perceptron.computePerceptronOutput(np.array([1,1]))
print or_perceptron.output

# Implementing Perceptrons to learn other functions (A ^ ~B)
custom_perceptron = perceptron(num_inputs=2, weight_vector=[-0.4,0.5,-1])

custom_perceptron.computePerceptronOutput(np.array([0,0]))
print custom_perceptron.output
custom_perceptron.computePerceptronOutput(np.array([0,1]))
print custom_perceptron.output
custom_perceptron.computePerceptronOutput(np.array([1,0]))
print custom_perceptron.output
custom_perceptron.computePerceptronOutput(np.array([1,1]))
print custom_perceptron.output

# Implment XOR function using the custom perceptron defined above
xor_lyr11 = perceptron(num_inputs=2, weight_vector=[-0.4,0.5,-1])
xor_lyr12 = perceptron(num_inputs=2, weight_vector=[-0.4,-1,0.5])
xor_lyr21 = perceptron(num_inputs = 2, weight_vector = [-0.3,0.5,0.5])

xor_lyr11.computePerceptronOutput(np.array([0,0]))
xor_lyr12.computePerceptronOutput(np.array([0,0]))
xor_lyr21.computePerceptronOutput(np.array([xor_lyr11.output, xor_lyr12.output]))
print xor_lyr21.output

xor_lyr11.computePerceptronOutput(np.array([0,1]))
xor_lyr12.computePerceptronOutput(np.array([0,1]))
xor_lyr21.computePerceptronOutput(np.array([xor_lyr11.output, xor_lyr12.output]))
print xor_lyr21.output

xor_lyr11.computePerceptronOutput(np.array([1,0]))
xor_lyr12.computePerceptronOutput(np.array([1,0]))
xor_lyr21.computePerceptronOutput(np.array([xor_lyr11.output, xor_lyr12.output]))
print xor_lyr21.output

xor_lyr11.computePerceptronOutput(np.array([1,1]))
xor_lyr12.computePerceptronOutput(np.array([1,1]))
xor_lyr21.computePerceptronOutput(np.array([xor_lyr11.output, xor_lyr12.output]))
print xor_lyr21.output
