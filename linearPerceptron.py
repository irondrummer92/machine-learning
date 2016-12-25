
# Random to intialize random perceptron weights (if required)
# Numpy for vectors,matrices and dot products
import random as rd
import numpy as np
import pandas as pd


# Defining a perceptron
#---------------------------------------------------------------------------------------#
class linearPerceptron(object):
    
    # Perceptron starts off with random weight vectors if not provided by default
    # Number of inputs must be equal to number of weights
    # Some debugging variables have also been captured
    
    def __init__(self, num_inputs, weight_vector = None):
        # Defining the initial perceptron
        self.num_inputs = num_inputs
        if(weight_vector == None):
            self.weights = [rd.random() for i in range(num_inputs)]
        else:
            if(len(weight_vector) != num_inputs):
                raise Exception("Weight vector size invalid. Must be #Inputs + 1")
            else:
                self.weights = weight_vector
        self.weights = np.array(self.weights)
        self.output = None
        self.iterError = []
        self.weightUpdates = []  
        self.minError = None
        self.runIters = 0

    # Perceptron outputs are a dot product between the inputs and the outputs
    
    def computePerceptronOutput(self, input_vector):
        # Computes the output as a combination of inputs
        if(type(input_vector) is np.array):
            print "Input not a numpy array"
        else:
            if(self.weights.size == input_vector.size):   
                self.output = np.dot(self.weights, input_vector)
            else:
                print "Inputs provided is of invalide size"
    
    
    # Error computation happens based on the formula
    # E = 1/2 * sum((td - od)^2)
    def computeError(self,input_matrix, target_vector):
        # Compute errors in the current weight training
        
        # Output vector corresponds to perceptron output for current set of weights
        outputVector = []
        
        nTrain = len(input_matrix) # Size of training input
        
        # Each input vector is passed through the perceptron
        for input_vector in input_matrix:
            self.computePerceptronOutput(input_vector)
            outputVector = np.append(outputVector,self.output)
        
        if(len(target_vector) != nTrain):
            raise Exception("Training Vector provided is not of the same size as inputs")

        # Computation of (td - od) - The absolute error
        error_vector = np.subtract(target_vector,outputVector)
        # Squaring the error
        error_squared = [x * x for x in error_vector]
        # SSE for the input dataset
        totalError = sum(error_squared) * 0.5
        
        # Compute updateSizes for each iteration
        # deltaw = sum((td - od) * xid) over d in D
        updateSize = np.dot(error_vector,input_matrix)
        return totalError,updateSize
    
    def trainBatchGradientDescent(self, input_matrix, target_vector, alpha = 0.01, nIter = 100):
        # Batch Gradient Descent is performed on the linear Perceptron given
        # the input matrix and a target vector to be learnt at learning Rate (alpha)
        # The batch descent is by default set to run only 100 iterations
        print "Training with Batch Gradient Descent....! Plz Wait..!"
        
        # Input matrix is number_of_inputs X no. of training examples
        self.weightUpdates = np.append(self.weightUpdates,self.weights)
        nInput = self.num_inputs
        
        # Minerror for the initial iteration
        self.minError,updateSizes = self.computeError(input_matrix = input_matrix,target_vector = target_vector)
        self.iterError = np.append(self.iterError,self.minError)
        
        # Convergence flag
        converged = 0
        
        # When nIter runs out without convergence, the current weights are considered final
        for i in range(nIter):
            outputVector = np.array([])
            
            totalError,updateSizes = self.computeError(input_matrix,target_vector)
            
            # Deltaweights for each weight
            # alpha (or eta) is the learning rate
            deltaWeights = alpha * updateSizes
            
            # UpdateWeights
            oldWeights = self.weights
            newWeights = np.add(self.weights,deltaWeights)
            self.weights = newWeights
            
            # After update, if the errors are not reducing, convergence is confirmed
            totalError,updateSizes = self.computeError(input_matrix,target_vector)
            
            if(totalError >= self.minError):
                converged = 1
                self.runIters = i
                self.weights = oldWeights
                print "Converged.....!"
                print self.weights
                return
            else:
                self.minError = totalError
                self.weights = newWeights
                self.weightUpdates = np.append(self.weightUpdates,self.weights)
                self.iterError = np.append(self.iterError,self.minError)

        
        print "Ran out of iterations without converging......! Tough luck matey...!"
        print self.weights          

#----------------------------------------------------------------------------------------------#  

# Generate training examples to learn the function
# x1 + 2 * x2 > 2

# Generating random inputs to the perceptron and bifurcating into the train and test datasets
nInputs = 250

x1 = [rd.random() for i in range(nInputs)]
meanx1 = np.mean(x1)
sdx1 = np.std(x1)
x1 = [round((x-meanx1)/sdx1,3) for x in x1]
x1 = np.array(x1)

x2 = [rd.random() for i in range(nInputs)]
meanx2 = np.mean(x2)
sdx2 = np.std(x2)
x2 = [round((x-meanx2)/sdx2,3) for x in x2]
x2 = np.array(x2)

inputMatrix = []
outputVector = []

for i in range(nInputs):
    inputMatrix.append([x1[i],x2[i]])
    outputVector.append(1 if ((x1[i] + 2 * x2[i] - 2) > 0) else 0)
    
inputMatrix = np.array(inputMatrix)
outputVector = np.array(outputVector)
#--------------------------------------------------------------------------------------------------#

# Train and test datasets
trn = np.random.randn(250) < 0.8

train = inputMatrix[trn]
trainOutput = outputVector[trn]
print len(train)
print len(trainOutput)

test = inputMatrix[~trn]
testOutput = outputVector[~trn]
print len(test)
print len(testOutput)

#--------------------------------------------------------------------------------------------------#

# Initialize a perceptron and run batch gradient descent
lp1 = linearPerceptron(num_inputs=2)
lp1.trainBatchGradientDescent(alpha=0.001,input_matrix=train,target_vector=trainOutput,nIter=100)

# Print all diagnostics 
print lp1.iterError
print lp1.runIters

print lp1.computeError(input_matrix=test,target_vector=testOutput)[0]
