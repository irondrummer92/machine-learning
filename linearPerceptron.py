
# Random to intialize random perceptron weights (if required)
# Numpy for vectors,matrices and dot products
import random as rd
import numpy as np
import pandas as pd


# Defining a perceptron
#---------------------------------------------------------------------------------------#
# Defining a perceptron

class linearPerceptron(object):
    
    def __init__(self, num_inputs, weight_vector = None):
        # Defining the initial perceptron
        self.num_inputs = num_inputs
        if(weight_vector is None):
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
        self.converged = 0

    
    def computePerceptronOutput(self, input_vector):
        # Computes the output as a combination of inputs
        if(type(input_vector) is np.array):
            print "Input not a numpy array"
        else:
            if(self.weights.size == input_vector.size):   
                self.output = np.dot(self.weights, input_vector)
            else:
                print "Inputs provided is of invalide size"
    
                
    def computeError(self,input_matrix, target_vector):
        # Compute errors in the current weight training
        
        outputVector = []
        
        nTrain = len(input_matrix) # Size of training input
        
        for input_vector in input_matrix:
            self.computePerceptronOutput(input_vector)
            outputVector = np.append(outputVector,self.output)
        
        if(len(target_vector) != nTrain):
            raise Exception("Training Vector provided is not of the same size as inputs")

        error_vector = np.subtract(target_vector,outputVector)
        error_squared = [x * x for x in error_vector]
        totalError = sum(error_squared) * 0.5
        updateSize = np.dot(error_vector,input_matrix)
        return totalError,updateSize
    
    
    def trainGradientDescent(self, input_matrix, target_vector, alpha = 0.01, nIter = 100, method = "batch"):
        
        
        # Batch Gradient Descent is performed on the linear Perceptron given
        # the input matrix and a target vector to be learnt at learning Rate (alpha)
        # The batch descent is by default set to run only 100 iterations
        if(method=="batch"):
            print "Training with Batch Gradient Descent....! Plz Wait..!"

            # Input matrix is number_of_inputs X no. of training examples

            self.weightUpdates = np.append(self.weightUpdates,self.weights)

            nInput = self.num_inputs

            self.minError,updateSizes = self.computeError(input_matrix = input_matrix,target_vector = target_vector)
            self.iterError = np.append(self.iterError,self.minError)

            for i in range(nIter):

                totalError,updateSizes = self.computeError(input_matrix,target_vector)

                deltaWeights = alpha * updateSizes

                # UpdateWeights
                oldWeights = self.weights
                newWeights = np.add(self.weights,deltaWeights)
                self.weights = newWeights

                totalError,updateSizes = self.computeError(input_matrix,target_vector)

                if(totalError >= self.minError):
                    self.converged = 1
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
    
        # For stochastic gradient descent
        elif(method == "incremental"):
            
            print "Training with Stochastic Gradient Descent....! Plz Wait..!" 
            
            self.weightUpdates = np.append(self.weightUpdates,self.weights)

            nInput = self.num_inputs
            nTrain = len(input_matrix)
            
            self.minError,updateSizes = self.computeError(input_matrix = input_matrix,target_vector = target_vector)
            self.iterError = np.append(self.iterError,self.minError)
            
            oldWeights = self.weights
            
            # Run through iterations
            for i in range(nIter):
                
                # Run through all training examples
                for j in range(nTrain):
                    
                    input_vector = input_matrix[j]
                    current_output = target_vector[j]
                    
                    newWeights = self.weights
                    
                    self.computePerceptronOutput(input_vector)
                    deltaWeights = alpha * (current_output - self.output) * input_vector

                    newWeights = np.add(newWeights,deltaWeights)
                    
                    self.weights = newWeights

                totalError,updateSizes = self.computeError(input_matrix,target_vector)

                if(totalError > self.minError):
                    self.converged = 1
                    self.runIters = i
                    self.weights = oldWeights
                    self.minError = totalError
                    self.iterError = np.append(self.iterError,totalError)
                    self.weightUpdates = np.append(self.weightUpdates,self.weights)
                    print "Converged.....!"
                    print self.weights
                    return

                else:
                    self.minError = totalError
                    self.weightUpdates = np.append(self.weightUpdates,self.weights)
                    self.iterError = np.append(self.iterError,self.minError)

            print "Ran out of iterations without converging......! Tough luck matey...!"
            print self.weights
        
        
        else:
            
            print "No valid method provided...! Must be either 'stochastic' or 'batch'"
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

# Training lp1 with Batch Gradient Descent
lp1 = linearPerceptron(num_inputs=2,weight_vector=np.array([0,0]))
lp1.trainGradientDescent(alpha=0.001,input_matrix=train,target_vector=trainOutput,nIter=100,method="batch")


# Training lp2 with Incremental Gradient Descent
lp2 = linearPerceptron(num_inputs=2)
lp2.trainGradientDescent(alpha=0.001,input_matrix=train,target_vector=trainOutput,nIter=100,method="incremental")

print lp1.iterError
lp1.runIters

print lp2.iterError

lp2.runIters
