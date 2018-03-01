'''
Created on Feb 23, 2018
'''
import numpy as np
import random
import math
import csv
random.seed()


# Here are the settings for the neural network
def setParameters():
    global layerCount, layerSize, maxEpochs, learningRate, kSplit, activationFunc, momentumAlpha, useMomentum, useSoftmax, dataType, groupsOf64, splitPoint
    global rProp, rPropInc, rPropDec
    
    #WHAT DATA SET ARE WE WORKING WITH?
    #Set to "leaf" if using leaf data
    #Set to "number" if using handwritten digit data
    dataType = "leaf"
    
    maxEpochs = 100 #This is the maximum number of maxEpochs until a network converges
    layerCount = 3 #The number of layers in the network
    layerSize = [192,60,99] #The number of nodes in each layer, a 2 hidden layer network might look like [64,30,25,10] (number takes 64 inputs, leaf takes 64,128,192 input nodes)
    learningRate = 0.3
    
    kSplit = 8 # I use a kfold method to split data into training and testing sets. This is the number of splits. ***This should be the same as splitPoint below when using leaf***
    
    #For leaf data only
    #the i'th example per leaf classification to split between training and testing (ie if splitPoint = 8, 8 examples go to training and 2 go to testing. If splitPoint = 7, 7 train / 3 test)
    splitPoint = 8
    
    useMomentum = True
    momentumAlpha = 0.2
    
    #If false, normal backpropagation is used
    rProp = False
    rPropInc = 1.2
    rPropDec = 0.5
    
    activationFunc = "sigmoid" #If "sigmoid", hidden layers use sigmoid.. If "tan", the hidden layers use tan.
    useSoftmax = False #If "softmax", softmax is used (only used for output layer)
    
    #used to say how many of the groups we want to use for leaf data set
    groupsOf64 = 3 #the number of those groups of 64 data pieces to incorporate (margin, shape, texture)
    

def loadDigitData():
    images = []

    for digit in range (10):
        with open('a1digits/digit_train_' + str(digit) + '.txt', newline='') as inputfile:
            for row in csv.reader(inputfile):
                images.append([digit] + row)
    
    return images


def loadDigitTestData():
    images = []

    for digit in range (10):
        with open('a1digits/digit_test_' + str(digit) + '.txt', newline='') as inputfile:
            for row in csv.reader(inputfile):
                images.append([digit] + row)
    
    return images


def loadLeafData(groupsOf64):
    leaves = [] #contains all the leaves in the set
    leafTrainingSet = [] #used to return each training and testing set
    leafTestingSet = [] #used to return each training and testing set
    
    numOfElements = groupsOf64 * 64 + 2
    
    #Open Data file
    with open('train.csv') as inputfile:
        for row in csv.reader(inputfile):
            leaves.append(row[:numOfElements])
            
    #remove first item as it is a title row
    leaves.pop(0)
    
    #Sort the list by species name
    leaves.sort(key=lambda x : x[1])
    
    #set their ID to the current species
    speciesNumber = 0;

    currSpecies = leaves[0][1]
    for l in leaves:
        if l[1] != currSpecies:
            currSpecies = l[1]
            speciesNumber+=1

        l[0] = speciesNumber
        #print (l)
        
    #split the leaf classes evenly, 8 training, 2 testing
    for i in range (0,len(leaves),10):
        for j in range(splitPoint):
            leafTrainingSet.append(leaves[i+j])
        for j in range(splitPoint,10):
            leafTestingSet.append(leaves[i+j])

    return [leafTrainingSet,leafTestingSet]


#Creates the neural network architecture
#NOTE some initial sizings are not correct and are replaced with correct sizing within the actual algo. This is in part for readability
def createMatrices():
    global layersS, layersZ, layersW, layersZDer, layersDelta, layerSize
    global layerCount, useMomentum, prevChangeW
    
    #S = sum, Z = squashed sum, zDer = derivative of squashing func, W = weights, Delta = Error*Derivative...
    layersS = [None] * layerCount
    layersZ = [None] * layerCount
    layersZDer = [None] * layerCount
    layersW = [None] * layerCount
    layersDelta = [None] * layerCount
    prevChangeW = [None] * layerCount #Used for momentum if momentum is true
    
    
    layersZ[0] = np.empty((1,layerSize[0])) #IN MATRIX contains the in values
    
    #hidden layers
    for l in range (1, layerCount):
        
        layersW[l] = np.random.random_sample((layerSize[l-1], layerSize[l])) - 0.5 #Stores weights from i to j
        #Initialize momentum matricies with the same shape as weights
        if (useMomentum):
            prevChangeW[l] = np.zeros((layersW[l].shape))
            
        layersS[l] = np.empty((1, layerSize[l])) #Stores summed values, is the result of dot(inMatrix, w1Matrix)
        layersZ[l] = np.empty((1, layerSize[l])) #Squashes values

        layersZDer[l] = np.empty((1, layerSize[l])) #Derived values
        layersDelta[l] = np.empty((1, layerSize[l])) #Delta values
            
    
#Helper Functions
def squash(x,func):
    if (func == "sigmoid"): 
        return 1 / (1 + pow(math.e, (-1 * x)))
    elif (func == "tan"):
        return (pow(math.e,x) - pow(math.e,(-1*x))) / (pow(math.e,x) + pow(math.e,(-1*x)))


def squashDer(x, func):
    if (func == "sigmoid"):
        return squash(x, func) *  (1-squash(x,func))
    elif (func == "tan"):
        return 1 - pow(squash(x,func), 2)

        
def calcZ(sMatrix, zMatrix, func):
    if (func == "softmax"):
        zMatrix = np.exp(sMatrix) / np.sum(np.exp(sMatrix), axis=1)
        return zMatrix
    else:
        for i,x in enumerate(sMatrix[0]):
            zMatrix[0,i] = squash(x,func)

def calcZDer(zMatrix, zDerMatrix, func):
    for i,x in enumerate(zMatrix[0]):
        zDerMatrix[0,i] = squashDer(x,func)
        
    
#Neural Network Functions
def forwardPass():
    for l in range (1, layerCount-1):
        layersS[l] = np.dot(layersZ[l-1], layersW[l])
        calcZ(layersS[l], layersZ[l], activationFunc)
        calcZDer(layersS[l], layersZDer[l], activationFunc)
        
    ##Need to use softmax as activation for last layer when softmax = true
    l = layerCount-1
    layersS[l] = np.dot(layersZ[l-1], layersW[l])
    
    if (useSoftmax == True):
        layersZ[l] = calcZ(layersS[l], layersZ[l], "softmax")
    else:
        calcZ(layersS[l], layersZ[l], activationFunc)
        

def errorProp(expected):
    layersDelta[layerCount-1] = np.subtract(layersZ[layerCount-1], expected).T
    
    for l in range (layerCount-1, 1, -1):
        layersDelta[l-1] = np.multiply(layersZDer[l-1].T, np.dot(layersW[l], layersDelta[l]))
            
    
def weightAdjust():    
    for l in range (0, layerCount-1):
        
        changeW = learningRate * (np.dot(layersDelta[l+1], layersZ[l])).T
        
        if (useMomentum):
            deltaMomentum = momentumAlpha * prevChangeW[l+1]
            layersW[l+1] = np.subtract(layersW[l+1], (changeW + deltaMomentum))
            prevChangeW[l+1] = (changeW + deltaMomentum)
        
        else:
            layersW[l+1] -= changeW    
            

def rPropAdjustWeights(bGradients):   
    global prevRWeights, prevrGradients
    currRWeights = prevRWeights[:] #just to initialize currRWeights
    
    #Calculate possible change of W from gradients sign
    
    #for each weightset, neuron, connections to neuron
    for layer in range(len(bGradients)):
        for i in range (len(bGradients[layer])):
            for j in range (len(bGradients[layer][i])):
                
                if ((prevrGradients[layer][i,j] * bGradients[layer][i,j]) > 0):
                    currRWeights[layer][i,j] = currRWeights[layer][i,j] * rPropInc
                
                elif ((prevrGradients[layer][i,j] * bGradients[layer][i,j]) < 0):
                    currRWeights[layer][i,j] = currRWeights[layer][i,j] * rPropDec
                    
                    bGradients[layer][i,j] = 0
                    
                else:
                    pass    #Do nothing, remains at prev value
                
                
    #We now have an updated set of delta weights
    #Apply actual change to W depending on gradients sign
                
                if (bGradients[layer][i,j] > 0):
                    layersW[layer+1][i,j] += (currRWeights[layer][i,j])
                
                elif (bGradients[layer][i,j] < 0):
                    layersW[layer+1][i,j] += (-1 * currRWeights[layer][i,j])

                else:
                    pass    #Do nothing, remains at prev value
    
    
    prevRWeights = currRWeights[:]
    prevrGradients = batchGradients[:]
    

#Calculates the gradients at each set of weights
#Returns a list of 2D arrays, each 2D array represents the gradients of its respective set of weights
def calculateGradients(batchGradients):
    currentGradients = [None] * (layerCount-1)

    for l in range (0, layerCount-1):
        
        if (useMomentum):
            
            changeW = np.dot((layersDelta[l+1] * -1), layersZ[l]).T
            deltaMomentum = momentumAlpha * prevChangeW[l+1]
            
            currentGradients[l] = (changeW + deltaMomentum)
            
            prevChangeW[l+1] = (changeW + deltaMomentum)
            
        else:
            currentGradients[l] = np.dot(layersDelta[l+1], layersZ[l]).T
        
        currentGradients[l] += batchGradients[l]

    return currentGradients


#Returns the error during validation runs (not during training)
def validationError(expected):    
    layersDelta[layerCount-1] = np.subtract(expected, layersZ[layerCount-1]).T


#Check for convergence
def checkConvergence(detailedResults):
    length = len(detailedResults)-1
    
    if (detailedResults[length] > 0.98):
        print ("greater than 98% convergence, we are probably memorizing at this point!")
        return True
    
    avg = 0
    for i in range (10):
        avg += detailedResults[length-i]
        
    avg = avg/10
    
    if (abs(detailedResults[length] - avg) < 0.002 and (abs(detailedResults[length] - detailedResults[length-1]) < 0.002)):
        print ("low amount of change")
        return True
    
    return False

#Returns the integer value of the results (ie. if the 35th output node is highest, it will return '34')
def identifyOutput():
    
    if (dataType == "number"):
        #get the highest output node value, that is used as most likely digit
        max = layersZ[layerCount-1][0,0]
        position = 0
        for i,output in enumerate(layersZ[layerCount-1][0]):
            if (output > max):
                max = output
                position = i
        
        return position
    
    #must return int val of the outputs
    elif (dataType == "leaf"):
        max = layersZ[layerCount-1][0,0]
        position = 0
        for i,output in enumerate(layersZ[layerCount-1][0]):
            if (output > max):
                max = output
                position = i

        return position


#Sets the input layer and expected output to the given data piece
def setInput(dataPiece):

    if (dataType == "number"):
        #Set inputs
        for x in range (1, len(dataPiece)):
            layersZ[0][0,x-1] = dataPiece[x]
            
        #Set expectedOutcome
        for x in range (layerSize[layerCount-1]):
            if (x == dataPiece[0]):
                    expected[0,x] = 1
            else:
                expected[0,x] = 0
                
    elif (dataType == "leaf"):
        #Set inputs
        for x in range (2, len(dataPiece)):
            layersZ[0][0,x-2] = dataPiece[x]
            
        #Set expectedOutcome
        expected[0] = np.zeros(len(expected[0]))
        expected[0,dataPiece[0]] = 1
        
                
            
#For personal result exporting, writes historical accuracy measurements to a csv file  
def exportResults():
    
    file.write("Run: " + str(currentRun))
        
    for accuracy in range(len(detailedResults)):
        file.write(str(detailedResults[accuracy]) + '\n')
            
    
    for result in range(len(finalTestResults)):
        file2.write(str(finalTestResults[result]) + '\n')
            
    
    print("---------------")
    print ("RESULTS ADDED TO: runResults.csv")
    print ("FINAL TEST RESULT ADDED TO: runFinalResults.csv")



#initialize our network, create the structure and load/shuffle the data
def initializeNetwork():
    global detailedResults, finalTestResults, batchGradients, expected, converged, trainingDataSet, testingDataSet, prevRWeights, prevrGradients
    
    #used for file saving at the end of a network run
    detailedResults = [] #Contains the detailed accuracy and results while a network is being trained
    finalTestResults = [] #contains the final test result of a network

    #called to build out our network
    setParameters()
    createMatrices()
    expected = np.empty((1,len(layersZ[layerCount-1][0])))
    
    #initialize rprop variables if using rprop
    if (rProp):
        batchGradients = [None] * (layerCount-1)
        
        prevrGradients = [None] * (layerCount-1)
        for i in range(len(prevrGradients)):
            prevrGradients[i] = np.zeros((layersW[i+1].shape))
            
        prevRWeights = [None] * (layerCount-1)
        for i in range(len(prevRWeights)):
            prevRWeights[i] = 0.1 * np.ones((layersW[i+1].shape))
            
    #load our data and shuffle
    if (dataType == "leaf"):
        dataSets = loadLeafData(groupsOf64)
        trainingDataSet = dataSets[0]
        testingDataSet = dataSets[1]
    
    elif (dataType == "number"):
        trainingDataSet = loadDigitData()
        testingDataSet = loadDigitTestData()
        
    random.shuffle(trainingDataSet)
    random.shuffle(testingDataSet)
    
    converged = False
    
    
# Run the actual network, this runs until the network has converged. See following steps:
#
# 1) Every k runs, re-randomize our data
# 2) set validation set to current k subset, combine all remaining subsets to our training set
# 3) Test on the validation set and record network accuracy
# 4) Train the network on the training set
# 5) Epoch complete, increment counter
def runNetwork():
    global currentEpoch, converged, batchGradients
    global trainingDataSet
    currentEpoch = 0
    
    while (currentEpoch < maxEpochs and not(converged)):
        
        #reset our batchGradient numbers for rprop
        if (rProp):
            for i in range(len(batchGradients)):
                batchGradients[i] = np.zeros((layersW[i+1].shape))
        
        # STEP 1 - Randomize Data
        
        if (currentEpoch%kSplit == 0):
                
            dataSubsets = [[] for x in range(kSplit)] #reset our list of data subsets
            random.shuffle(trainingDataSet)
                
            #split into kSplit groups
            count = 0
            for i in range(0, len(trainingDataSet), int(len(trainingDataSet)/kSplit)): #actually define new subsets
                dataSubsets[count] = trainingDataSet[i:i + int(len(trainingDataSet)/kSplit)]
                count += 1
            
        # STEP 2 - Update Data Sets
        
        #assign each group to train/test depending on currentEpoch run
        trainingSet = []
        validationSet = []
        for i in range(kSplit):
            if (i != currentEpoch%kSplit): #if the current kfold is this set, dont add to the training set
                trainingSet += dataSubsets[i]  
        validationSet = dataSubsets[currentEpoch%kSplit]
        
        # STEP 3 - Check Accuracy
        
        totalError = 0.0
        correctClassifications = 0
        
        for dataPiece in validationSet:
                
            setInput(dataPiece)
            forwardPass()
            validationError(expected) #get resulting error
                
                
            #get total output layer error, and increment our correct classification counter if output is correct
            for errorVal in layersDelta[layerCount-1]:
                totalError += abs(errorVal[0])
            if (identifyOutput() == dataPiece[0]):
                correctClassifications += 1
        
        print ("currentEpoch: " + str(currentEpoch) + " Average error per output of currentEpoch: " + str(totalError/(len(validationSet))))
        print (str(correctClassifications) + " / " + str(len(validationSet)) + " correctly classified - " + str(round((correctClassifications/len(validationSet)),3)) + "% accurate")
        detailedResults.append(round((correctClassifications/len(validationSet)),3))
        
        # STEP 4 - Train
        
        #train on every piece in data set
        for dataPiece in trainingSet:
            
            setInput(dataPiece)
            forwardPass()
            errorProp(expected)#Error Calculation
            
            #Learning
            if (rProp): #rprop runs in batch, collect the batch here
                batchGradients = calculateGradients(batchGradients)
            else: #otherwise we are doing backprop, update weights immediately
                weightAdjust()
            
        if (rProp): #apply entire batch of training here
            rPropAdjustWeights(batchGradients)
    
        # STEP 5 - Epoch Done
        
        currentEpoch += 1
            
        if (currentEpoch > 20): #Dont start checking for convergence until after a set number of maxEpochs
            converged = checkConvergence(detailedResults)
         

# runs the final test on the network, using the test file instead of training file
# with the leaf data, the testing set has no answers, so we will use the training file instead
def finalTest():
    global testingDataSet
    
    random.shuffle(testingDataSet)
    
    totalError = 0.0
    correctClassifications = 0
    
    for dataPiece in testingDataSet:
                 
        setInput(dataPiece)
        forwardPass()
        validationError(expected)#get resulting error
                 
        for errorVal in layersDelta[layerCount-1]:
            totalError += abs(errorVal[0])
        if (identifyOutput() == dataPiece[0]):
            correctClassifications += 1
    
    
    print("")
    print ("MAIN RESULTS FOR NETWORK: ")
    print ("currentEpoch: " + str(currentEpoch) + " Average error per output of currentEpoch: " + str(totalError/(len(testingDataSet))))
    print (str(correctClassifications) + " / " + str(len(testingDataSet)) + " correctly classified - " + str(round((correctClassifications/len(testingDataSet)),3)) + "% accurate")
    print("")
    print("")
    finalTestResults.append(round((correctClassifications/len(testingDataSet)),3))
    print (detailedResults)
    print (finalTestResults)
    exportResults()
    

#Start the program
file = open('runResults.csv', 'w')
file2 = open('runFinalResult.csv','w')

initializeNetwork()
file.write("30 Runs - layers: " + str(layerSize).replace(',', '-') + "   UseRProp: " + str(rProp) + "   activationFunc: " + activationFunc)

for run in range(3):
    print ("RUN NUMBER: " + str(run))
    initializeNetwork()
    runNetwork()
    finalTest()
    
file.close()
file2.close()