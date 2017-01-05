import numpy as np
import random as rd

# Lookup table of activation functions
actLookup = {"sigmoid":lambda x: 1/(1 + (np.exp(-1 * x))),
                         "linear":lambda x: x}
                         
class neuron(object):

    # Defining a neuron with n inputs and one output
    def __init__(self, numInputs, actFun = "sigmoid"):
        
        # Intialize with random weights
        self.numInputs = numInputs
        self.weights = [rd.random() for i in (range(numInputs + 1) )]
        self.weights = np.array(self.weights)
        
        if(actFun not in ('linear','sigmoid','tanh')):
            raise Exception("Unknown activation function. Input either sigmoid or tanh or linear")

        self.actFun = actLookup[actFun]
        self.inputs = None
        self.output = None
    
    def computeOutput(self,inputVector):
        
        self.inputs = inputVector
        
        if(not(isinstance(inputVector, np.ndarray))):
            inputVector = np.array(inputVector)
        
        # Size of inputs (Say for batch learning)
        batchSize, vectorSize = inputVector.shape()
        
        # Transpose the input vector and add a vector of 1s
        # Vector of 1s serves as input for the bias neurons
        inputVector = inputVector.T
        inputVector = np.append(np.ones(batchSize).reshape(1,2),inputVector, axis = 0)
        
        # Propagate the input and use activation function on the output        
        propOut = np.dot(self.weights, np.append(1,inputVector))
        
        self.output = self.actFun(propOut)
        # Return the output to a function that can pass it on to next layer
        return self.output
    

    def backward(self, trainTarget, delta, nextLayerWeights):
        # The backward function calculates the delta by neuron
        
        if(not(isinstance(trainTarget,np.ndarray))):
            trainTarget = np.array(trainTarget)
        
        # Compute the backward propagation error term
        if(delta is None):
            # If no delta has been passed, this neuron is the output neuron
            # Applying the formula delta = o(1-o)(t - o)
            errorTerm = np.subtract(trainTarget, self.output)
            outputInvert = np.array([(1 - op) for op in self.output])
            term1 = np.multiply(errorTerm, self.output)
            term2 = np.multiply(outputInvert, term1)
            self.delta = np.multiply(term2, self.weights)
            return self.delta
        else:
            # Since delta is not zero, this neuron is a hidden neuron
            outputInvert = np.array([(1 - op) for op in self.output])
            
        
    def updateWeights(self, weightUpdate):
        
        # weightUpdates are usually applied during backpropagation
        # Weight updates are applied all n + 1 input lines
        self.weights = np.add(self.weights,weightUpdate)
        
class neuralNetworkLayer(object):
    
    # Defining a layer of neurons with a set of inputs fed into each neuron and one output from each of them
    def __init__(self, numInputs, layerSize, actFun = 'sigmoid'):
        
        # Number of inputs and size of layer
        self.numInputs = numInputs
        self.layerSize = layerSize
        
        # If a vector of activation functions is defined, it needs to be equalt to the size of layer
        if(isinstance(actFun,list)):
            if(len(actFun) != self.layerSize):
                raise Exception("Activation function vector is not same size as the network")
            else:
                self.actFun = actFun
        else:
            print "Single activation function assigned to all neurons"
            self.actFun = [actFun for i in range(layerSize)]
            
        self.neurons = [neuron(numInputs,actFun=self.actFun[i]) for i in range(self.layerSize)]
        self.neuronInputs = None
        self.neuronOut
        
    # Defining a forward function for the layer:
    
    def forward(self, inputVector):
        
        # Input vector is the input to each neuron as well as the layer as a whole
        
        if(not(isinstance(inputVector,np.ndarray))):
            self.input = np.array(inputVector)
        
        # Each neuron computes the output fromt the input vector
        self.output = [neuronUnit.computeOutput(inputVector) for neuronUnit in self.neurons]
        return self.output
    
    # Calling a backward propagation on the layer
    
    def backward(self, delta, y):
        
        if(delta is None):
            # Propagation rule for the last layer
            
            
        else:
            # Propagation rule for hidden layer
