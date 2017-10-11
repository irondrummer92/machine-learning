
# Libraries required
import numpy as np
import random as rd

def dataCenters(inputMatrix):

    """Center mean of each feature to 0"""
    # (x - mu)/sigma. Data is now centered at 0

    rows,columns = inputMatrix.shape

    # Calculate mean and variance for each input feature
    variableMeans = np.mean(inputMatrix, axis = 0).reshape(1,columns)
    variableSD = np.std(inputMatrix, axis = 0).reshape(1,columns)

    return variableMeans, variableSD

# Lookup table of activation functions and derivatives
actLookup = {"sigmoid":lambda x: 1/(1 + np.exp(-1 * x)),
             "tanh":lambda x: np.tanh(x),
            "relu": lambda x: np.maximum(0,x)}

derivativeLookup = {"sigmoid": lambda x: x * (1.-x),
               "tanh": lambda x: 1. - (np.tanh(x) ** 2),
              "relu": lambda x: np.ceil(x.clip(0,1))}

# Weight Initialization strategy
# Random multiplies the initializations by a small number
# Xavier 1/sqrt(numInputs)
# For relu's, initalization is as recommended by He et. al(2015)
weightLookup = {"random":lambda x:0.01,
               "xavier":lambda x:1./np.sqrt(x),
               "he.et.al":lambda x:np.sqrt(2./x)}

# Regularization error component,
# 0 if none, sum(x^2) if L2 and sum(|x|) if L1
regLookup = {"none":lambda x: 0,
            "l2": lambda x: 0.5 * (x ** 2).sum(),
            "l1" : lambda x:sum(abs(x)),
            "l1+l2":lambda x: 0.5 * (x ** 2).sum() + sum(abs(x))}

# Derivatives of regularization terms
regDerivLookup = {"none": lambda x:0,
                "l2": lambda x: x,
                "l1": lambda x: 1,
                 "l1+l2":lambda x: x + 1.}

# Momentum rate update
momentumLookup = {"none":0,
                  "momentum":lambda x:1,
                  "nag": lambda x: (1+x)}

class linearLayer(object):
    # A neural net layer with n neurons as defined by the user and an activation function for the layer
    # Init: Creates a new linear layer with weight init method,
    # layer size and number of inputs as inputs
    # Forward propagation method: Dot product of input and weights
    # Backward propagation: Dot product of delta from next layer with weights
    # Weight update: Based on the per input dot product of the inputs and weights,
    # weight gradient is calculated and Adam is used as learning method

    def __init__(self,numInputs, layerSize, weightInit = "random"):

        # Size of the layer and the number of inputs
        self.layerSize = layerSize

        # Incorporating the bias neuron
        self.numInputs = numInputs + 1
        self.inputMatrix = None

        # Numpy matrix of weights randomly initialized
        # Seed is set to ensure repeatability of weight initializations and hence convergence and outputs
        np.random.seed(seed=10000)
        self.weights = np.array(np.random.randn(self.layerSize,self.numInputs)) * \
        weightLookup[weightInit](self.numInputs)

        # Init backward propagation  and learning values (only Adam) for this layer
        self.delta = None
        self.m = np.zeros(self.weights.shape)
        self.velocity = np.zeros(self.weights.shape)


    # Defining the forward function to the layer
    def forward(self, inputMatrix):
        """Forward() forward propagates the inputs to a layer"""

        # Convert to numpy array if not passed as a numpy array
        if(not(isinstance(inputMatrix,np.ndarray))):
            inputMatrix = np.array(inputMatrix)

        # Get input matrix shape (m x n)
        rows, columns = inputMatrix.shape

        # Dot product of input matrix (with extra 1s for the bias neuron) with the weight matrix
        inputPadded = np.append(np.ones(rows).reshape(rows,1), inputMatrix, axis = 1)
        self.inputMatrix = inputPadded

        # Forward propagation is dot product of weights and input values
        self.output = np.dot(inputPadded, self.weights.T)
        return self.output

    # Defining the backward propagation function
    def backward(self, delta):
        """Backward takes the delta from next layer and passes it on the previous layer"""

        # Local copy of delta
        self.delta = delta

        # Delta to pass on to preceding layer has to be
        # matrix multiplication of these values by the weight matrix
        weightMatrix = self.weights[:,1:]

        # Delta passed back is the dot product of weights and deltas by
        deltaBack = self.delta.dot(weightMatrix)
        return deltaBack

    # Adam only update since it is the latest learning technique
    def updateWeights(self, learnRate,
                      regularize = "none", lambdaReg = 0.01,
                      beta1 = 0.9, beta2 = 0.999, eps = 1e-8, t = 1):
        """updateWeights() uses the inputs and the delta stored in each layer after forward
        and backward propagation to derive the weight update rule"""

        # Depending on the Lookup
        regDeriv = regDerivLookup[regularize]

        # The udpate rule for each element in the matrix is given by
        # wi = wi + sum_over_instances(delta for neuron * input i to neuron * learning rated)
        # Compute the weight updates for every data point and then add those updates
        rows,columns = self.inputMatrix.shape
        stepGradient = -1. * np.dot(self.inputMatrix.T,self.delta).T * (1/np.float(rows))

        # Regularization if applicable
        stepGradient = np.add(stepGradient, lambdaReg * regDeriv(self.weights))

        # First and second moments for Adam
        self.m = beta1 * self.m + (1. - beta1) * stepGradient
        self.velocity = beta2 * self.velocity + (1. - beta2) * (stepGradient ** 2)

        # Bias correction for Adam. THIS DOES NOT GET CARRIED OVER TO THE NEXT ITERATION
        biasM = self.m/(1. - beta1 ** t)
        biasVelocity = self.velocity/(1. - beta2 ** t)

        # Final Weight updates using Adam
        self.weights = np.add(self.weights, -learnRate * biasM / (np.sqrt(biasVelocity) + eps))


class activationLayer(object):
    # A neural net activation layer
    # It has the same methods as a linear layer but no weight updates are done on this layer
    # Forward only propagates inputs with activations performed on them
    # Backward propagates incoming delta by differentiating
    # d(activation) * delta by element
    
    def __init__(self, actFun = "sigmoid"):

        # Activation function
        self.actFun = actLookup[actFun]
        self.actDeriv = derivativeLookup[actFun]
        
    # Forward propagation is just activation of inputs
    def forward(self, inputMatrix):
        """Forward propagation is just passing through activation function"""
        
        # Propagation of inputs through activation
        # Saving inputs is not required since this layer 
        # does not have any input related weight updates
        self.output = self.actFun(inputMatrix)
        return self.output
    
    # Backward propagation is differential of inputs and delta from next layer
    def backward(self, delta):
        """Backward propagation multiplies delta from next layer with derivative"""
        
        # Delta for this layer is multiplication of outputs with delta
        return np.multiply(self.actDeriv(self.output), delta)
    
    def updateWeights(self, learnRate,
                      regularize = "none", lambdaReg = 0.01,
                      beta1 = 0.9, beta2 = 0.999, eps = 1e-8, t = 1):
        """Do nothing"""
        
        # Does nothing
        return

class activationLayer(object):
    # A neural net activation layer
    # It has the same methods as a linear layer but no weight updates are done on this layer
    # Forward only propagates inputs with activations performed on them
    # Backward propagates incoming delta by differentiating
    # d(activation) * delta by element

    def __init__(self, actFun = "sigmoid"):

        # Activation function
        self.actFun = actLookup[actFun]
        self.actDeriv = derivativeLookup[actFun]

    # Forward propagation is just activation of inputs
    def forward(self, inputMatrix):
        """Forward propagation is just passing through activation function"""

        # Propagation of inputs through activation
        # Saving inputs is not required since this layer
        # does not have any input related weight updates
        self.output = self.actFun(inputMatrix)
        return self.output

    # Backward propagation is differential of inputs and delta from next layer
    def backward(self, delta):
        """Backward propagation multiplies delta from next layer with derivative"""

        # Delta for this layer is multiplication of outputs with delta
        return np.multiply(self.actDeriv(self.output), delta)

    def updateWeights(self, learnRate,
                      regularize = "none", lambdaReg = 0.01,
                      beta1 = 0.9, beta2 = 0.999, eps = 1e-8, t = 1):
        """Do nothing"""

        # Does nothing
        return

class dropoutLayer(object):
    # Dropout layer for neural nets
    # It creates a binary mask to drop certain outputs from the previous layer
    # Forward propagation is multiplication of each element of layer input
    # with the binary mask created
    # Backward propagation is delta multiplied by the binary mask and passed back
    
    def __init__(self, layerSize, p = 0.5):
        
        # p for binary and layer size are saved
        self.p = p
        self.layerSize = layerSize
        
    # Forward propagation is applying binary mask to the inputs
    def forward(self, inputMatrix):
        
        # Create binary mask at every epoch
        self.binaryMask = np.random.choice([0,1], size = self.layerSize,p = [self.p, 1.- self.p])
        
        # Return self.output as multiplication of binary mask
        self.output = np.multiply(inputMatrix, self.binaryMask)
        return self.output
    
    # Backward propagation for same iteration is reused
    def backward(self, delta):
    
        # Using the created binary mask
        deltaback = np.multiply(delta, self.binaryMask)/self.p
        return deltaback
    
    def updateWeights(self, learnRate,
                      regularize = "none", lambdaReg = 0.01,
                      beta1 = 0.9, beta2 = 0.999, eps = 1e-8, t = 1):
        """Do nothing"""
        
        # Does nothing
        return

class neuralNet(object):

    # Defining a neural net. To start, the object requires

    def __init__(self, layers, actFun = "sigmoid", weightInit = "random"):

        self.layers = []

        for layerDesc in layers:

            # Check for layer type and create layer
            if(layerDesc["type"] == "linear"):
                # Layer creates linear combination
                self.layers.append(linearLayer(layerSize=layerDesc["layerSize"],
                                                            numInputs=layerDesc["numInputs"],
                                                            weightInit=weightInit))

            elif(layerDesc["type"] == "activation"):
                # Create activation layer
                self.layers.append(activationLayer(actFun=actFun))

            elif(layerDesc["type"] == "dropout"):
                # Create Dropout layer
                self.layers.append(dropoutLayer(layerSize=layerDesc["layerSize"],
                                                              p=layerDesc["p"]))
            else:
                "Print unknown layer type"

        # CV Error and Regularization loss initialization
        self.cvError = None
        self.regLoss= None
        self.classAccuracy = None

    # Compute MSE on the input
    def computeError(self, inputs, outputs):

        predictions = self.predict(inputs)
        numRows = inputs.shape[0]

        # Calculate deltas and sum squares
        delta = np.subtract(predictions,outputs)
        totalError = np.sum(delta**2)/numRows

        preds = np.array([np.round(arr) for arr in predictions])

        # Calculate prediction accuracy
        predClass = np.packbits(preds.astype("bool"))
        inClass = np.packbits(outputs.astype("bool"))

        # Calculating class prediction accuracy
        predAcc = [predClass[i] == inClass[i] for i in range(len(predClass))]
        totalAcc = sum(predAcc)/np.float(len(predClass))

        return totalError, totalAcc

    # Calculate loss due to regularization
    def computeRegularizationLoss(self, regularize, lambdaReg):
        # Calculate layerWise sums and get total
        regLossFunction  = regLookup[regularize]
        linLayers = [layer for layer in self.layers if isinstance(layer,linearLayer)]
        layerRegs = np.array([lambdaReg * regLossFunction(layer.weights) for layer in linLayers])
        return(layerRegs.sum())

    # Defining forward propagation operation
    # Forward method of each layer is invoked
    def predict(self, inputMatrix):

        """Prediction given current layer weights and input matrix"""

        for layer in self.layers:
            layerOut = layer.forward(inputMatrix)
            inputMatrix = layerOut

        self.output = layerOut
        return self.output

    # Defining the backprop operation
    def backProp(self, numEpochs, batchSize, learnRate,
                 trainInput, trainOutput, testInput,testOutput,
                 showStep = 1000, stepError = 100,
                 regularize = "none",lambdaReg = 0.01,
                 beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        """Training algorithm for the neural net"""


        if(not(isinstance(trainOutput,np.ndarray))):
            trainOutput = np.array(trainOutput)
        if(not(isinstance(trainInput,np.ndarray))):
            trainInput = np.array(trainInput)

        rows, columns = trainInput.shape


        # Initialize losses for the MSE and Regularization losses
        self.cvError = np.zeros(numEpochs/stepError * 2).reshape(numEpochs/stepError,2)
        self.regLoss = np.zeros(numEpochs/stepError).reshape(numEpochs/stepError,1)
        self.classAccuracy = np.zeros(numEpochs/stepError * 2).reshape(numEpochs/stepError,2)

        for i in range(numEpochs):

            # Run for a prespecified number of iterations
            if((i + 1) % showStep == 0):
                print i + 1

            if((i + 1) % stepError == 0):

                mseTrain, accTrain = self.computeError(trainInput, trainOutput)
                mseTest, accTest = self.computeError(testInput, testOutput)

                self.cvError[((i + 1) / stepError)-1] = [mseTrain,mseTest]

                self.classAccuracy[((i + 1) / stepError)-1] = [accTrain,accTest]

                self.regLoss[((i + 1) / stepError)-1] = self.computeRegularizationLoss(regularize = regularize,
                                                                                        lambdaReg = lambdaReg)

           # Pick a random sample from the trainInput and trainOutput
            # Updated weights based on the same. Sample size is to be of size batchSize

            randomIndices = np.random.choice(range(rows),size=batchSize)

            # Sample from trainInput and trainOutput
            batchTrain = trainInput[randomIndices,:].reshape(batchSize,columns)
            batchTest = trainOutput[randomIndices,:]

            # A forward pass through the network
            output = self.predict(batchTrain)

            # Iterate backwards through the layers to pass the deltas
            delta = np.subtract(batchTest,output)

            for layer in self.layers[::-1]:

                # Delta to be passed to the previous layer is computed
                delta = layer.backward(delta=delta)
                # Update weights as determined by the delta gradient
                layer.updateWeights(learnRate=learnRate,
                                    regularize = regularize, lambdaReg = lambdaReg,
                                    beta1 = beta1, beta2 = beta2, eps = eps, t = i + 1)

