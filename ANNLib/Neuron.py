'''
Created on Feb 3, 2017

@author: Matth
'''
import random
import math

class Neuron():
    '''
    classdocs
    '''
    outputVal = 0.0
    outputValDer = 0.0
    errorVal = 0.0
    e = 0.0
    bias = 0.0
    prevMomentum = 0.0
    useSig = True
    expectedOut = 0.0
    

    def __init__(self, func):
        if  func == "tan":
            self.useSig = False
            print ("triggered")
        else:
            self.useSig = True
    
    'neuron layer is the list of neurons in the layer before the current layer'
    'layer and currentN are the current layer and node we are summing for'
    'weightlist is the list of weights'
    'we iterate through weightlist by the current layer, then current node, then its specific weight list of connections'
    '   note that those weights are of the previous layer to the current layer'
    def summation(self, neuronLayer, weightList, connectionLayer, currentN):
        sum = 0.0
        for count,neuron in enumerate(neuronLayer):
            sum += (float(neuron.outputVal) * weightList[connectionLayer][count][currentN])
        self.e = sum
    
    def squash(self, ):
        if (self.useSig == True):
            self.sigmoid()
        else:
            self.tan()
            
    def derivedSquash(self):
        if (self.useSig == True):
            self.derivedSigmoid()
        else:
            self.derivedTan()
        
    def sigmoid(self):
        x =  1 / (1 + pow(math.e, (-1 * self.e)))
        self.outputVal = x
    
    def derivedSigmoid(self):
        x = pow(math.e, self.e) / pow((pow(math.e, self.e) + 1), 2)
        self.outputValDer = x
        
    def tan(self):
        pass
    
    def derivedTan(self):
        pass