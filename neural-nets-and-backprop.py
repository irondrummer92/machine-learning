#----------------------------------Neural Nets v0.1---------------------------#

import numpy as np
import random as rd

#-------------------------Function definitions--------------------------------#

# Computation of the sigmoid function
def sigmoid(inputSignal):
    return 1/(1 + (np.exp(-1 * inputSignal)))

# Wrapper for the numpy dot product function
def dotProduct(numpyVector1,numpyVector2):
    if not(isinstance(numpyVector1,np.ndarray)):
        numpyVector1 = np.array(numpyVector1)
    if not(isinstance(numpyVector2,np.ndarray)):
        numpyVector2 = np.array(numpyVector2)
    if numpyVector1.size != numpyVector2.size:
        raise Exception("Dimensions do not match for dot product")
    else:
        return np.dot(numpyVector1,numpyVector2)

#------------------------------------------------------------------------------#

#--------------------------------------Neuron Object----------------------------#
class neuron(object):

    # Defining a sigmoid neuron with n inputs and one output
    def __init__(self, num_inputs, actFun = "sigmoid"):
        # Intialize with random weights
        self.num_inputs = num_inputs
        self.weights = [rd.random() for i in (range(num_inputs + 1) )]
        self.weights = np.array(self.weights)
        self.actFun = actFun
        self.output = None
    
    def forwardPropagate(self,inputVector):
        
        # Propagate the input and use activation function on the output        
        propOut = dotProduct(self.weights, np.append(1,inputVector))
        if self.actFun == "sigmoid":
            self.output = sigmoid(propOut)
        else:
            print "Activation function not found"
        
        # Return the output to a function that can pass it on to next layer
        return self.output
        
    def adjustWeights(self, weightUpdate):
        
        # weightUpdates are usually applied during backpropagation
        # Weight updates are applied all n + 1 input lines
        if(self.weights.size != weightUpdate.size):
            if not(isinstance(weightUpdates,np.ndarray)):
                weightUpdate = np.array(weightUpdate)
            self.weights = np.add(self.weights,weightUpdate)
            
#---------------------------------------------------------------------------------#


#----------------------------------Neural Net object------------------------------#
class neuralNet(object):
    
    # Defining a neural network that takes in the following inputs
    # Number of neurons in each layer as a vector. An 8x3x1 neural net can be initialized with n = [8,3,1]
    # Weights are randomly initialized for all neurons. Note: Later planning to add matrix inputs for custom weights
    # Methods include 
    # 1. Forward propagation
    # 2. Backprop training algorithm
    
    def __init__(self, n, actFun = "sigmoid"):
        
        if(not(isinstance(n,np.ndarray))):
            n = np.array(n)
            
        self.n = n
        
        # Number of layers excluding the input layer
        numLayers = n.size - 1
        
        # Check if n is of the right size (>=2)
        if(n.size < 2 or n.ndim > 1):
            raise Exception("Input vector is not the right size. Specify a 1D vector with at least 2 elements")
        
        # Defining a neural net based on n
        # Each layer has number of inputs = neurons in (n-1 layer)
        # Can list comprehension be used here?
        print "Generating Neural Net..!"
        
        # Empty net soon to be populated with randomly initialized neurons
        self.net = []
        
        for i in range(numLayers):
            numInputs = n[i]
            numNeurons = n[i+1]
            neuronLayer = [neuron(numInputs, actFun) for x in range(numNeurons)]
            self.net.append(neuronLayer)
            
        self.output = [rd.random() for x in range(n[-1])]
        
        print "Generated...!"
        
    
    # Defining the forward propagation algorithm for each layer
    # Output of each neeuron is a dot product of each neuron's weight and the input
    # The dot product is processed by the activation function to provide the final output
    def forwardPropagate(self,inputVector):
        
        # The input vector is assumed to be the same size as the expect inputs for the neuron
        # If not the dot product wrapper function dotProduct() raises an exception
        # For each neuron layerwise, this function propagates the input vector by calling
        # the forwardPropagate() of each neuron object
        
        layerInput = inputVector
        
        for layer in self.net:
            layerOutput = [nrn.forwardPropagate(layerInput) for nrn in layer]
            layerInput = layerOutput
    
    def getOutputs(self):
        
        # Return output of each neuron in last layer
        return [nrn.output for nrn in self.net[-1]]
    
    def getWeights(self):
        
        # Return weights of each neuron layerwise as a matrix
        return [[nrn.weights for nrn in layer] for layer in self.net]
        
    def backprop(self,trainData,trainTarget, learningRate, method = "stochastic"):
        
        # The standard back prop algorithm that trains each neuron by
        # propagating the error at the output stagewise backwards
        # If t is the training target vector and o is the output observed:
        # The output layer weights are adjusted by the quantity:
    
        print "Backpropagating... Wait...!"
        
#---------------------------------------------------------------------------------#

nn1 = neuralNet([3,3,1])

nn1.forwardPropagate([1 for i in range(3)])
