import numpy as np
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # y er hvor mange neurons i dette laget, mens x er hvor mange i forrige

 

   
    def feedforward(self, a):
        #Går lag for lag og gjøre activation function på hver av inputsa hvor a da blir en vector med alle activasionsa
        
        for b, w in zip(self.biases, self.weights):
            #weights ganger activation pluss bias, sigmod blir gjort på hver og en neuron på grunn av numpy 
            a = sigmoid(np.dot(w,a)+ b)
            
        return a

network = Network([2, 3, 1])
print(network.biases)
print(network.weights)
print(network.feedforward(np.array([1, 2]).reshape(-1, 1)))



